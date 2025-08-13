import numpy as np
import elastica as ea
from elastica.boundary_conditions import FixedConstraint
from elastica.joint import get_relative_rotation_two_systems
from elastica.timestepper import extend_stepper_interface
from elastica.rigidbody import Sphere


class ConstantCurvatureBase(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
    ea.Connections,
):
    pass

from src.DataCallback import DataCallback, RigidSphereCallBack

class Sim:
    """
    A class to encapsulate the constant curvature rod simulation setup and execution.
    """
    
    def __init__(self, 
                 n_elem=30,
                 base_length=0.4,
                 base_radius=0.02,
                 density=1000,
                 youngs_modulus=1e5,
                 poisson_ratio=0.5,
                 damping_constant=1e-1,
                 dt=1e-4,
                 double_rod=True,
                 max_torque=5e-2,
                 mpc_dt=0.02,
                 with_targets=False
                 ):
        """
        Initialize the constant curvature simulation.
        
        Parameters:
        -----------
        n_elem : int
            Number of elements in each rod
        base_length : float
            Length of each rod
        base_radius : float
            Radius of each rod
        density : float
            Material density
        youngs_modulus : float
            Young's modulus
        poisson_ratio : float
            Poisson's ratio
        damping_constant : float
            Damping coefficient
        dt : float
            Time step
        double_rod : bool
            Whether to create two connected rods
        max_torque : float
            Maximum applied torque
        mpc_dt : float
            Model predictive control time step
        """
        
        # Store parameters
        self.n_elem = n_elem
        self.base_length = base_length
        self.base_radius = base_radius
        self.density = density
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        self.damping_constant = damping_constant
        self.dt = dt
        self.double_rod = double_rod
        self.max_torque = max_torque
        self.mpc_dt = mpc_dt
        
        # Computed parameters
        self.shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
        self.step_skip = mpc_dt / dt
        self.base_area = np.pi * base_radius ** 2
        
        # Rod directions
        self.direction = np.array([0.0, 0.0, -1.0])
        self.normal = np.array([0.0, 1.0, 0.0])
        
        # Initialize simulation components
        self.cc_sim = None
        self.rods = []
        self.targets = []
        self.with_targets = with_targets
        self.torque_objects = []
        self.callback_params = []
        self.target_callback_params = []
        self.timestepper = None
        self.do_step = None
        self.stages_and_updates = None
        
        # Setup the simulation
        self._setup_simulation()
    
    def add_targets(self):
        """Add target spheres to the simulation."""
        # Define target positions and colors
        target_positions = [
            np.array([0.5, 0.0, -0.5]),  # right on the XY plane
            np.array([-0.5, 0.0, -0.5]), # lfet
            np.array([0.0, 0.5, -0.5]),  # up
            np.array([0.0, -0.5, -0.5]), # down
        ]
        
        target_colors = ['red', 'blue', 'orange', 'purple']
        
        # Create targets
        for i, (position, color) in enumerate(zip(target_positions, target_colors)):
            target = Sphere(
                center=position,
                base_radius=0.05,
                density=1000,
            )
            # Store color as an attribute for visualization
            target.target_color = color
            target.target_id = i
            
            self.cc_sim.append(target)
            self.targets.append(target)

    def _setup_simulation(self):
        """Setup the complete simulation including rods, constraints, and forces."""
        
        # Initialize the simulation collection
        self.cc_sim = ConstantCurvatureBase()
        
        # Create rods
        self._create_rods()
        
        # Add targets if requested
        if self.with_targets:
            self.add_targets()
        
        # Apply constraints, forces, and damping
        self._apply_constraints_and_forces()
        
        # Connect rods if double_rod is True
        if self.double_rod:
            self._connect_rods()
        
        # Setup callbacks for data collection
        self._setup_callbacks()
        
        # Finalize the simulation
        self.cc_sim.finalize()
        
        # Extract torque objects for dynamic control
        self._extract_torque_objects()
        
        # Setup timestepper
        self._setup_timestepper()
    
    def _create_rods(self):
        """Create the rod(s) for the simulation."""
        
        # Rod 1
        start_rod_1 = np.array([0.0, 0.0, 0.0])
        rod_1 = ea.CosseratRod.straight_rod(
            self.n_elem,
            start_rod_1,
            self.direction,
            self.normal,
            self.base_length,
            self.base_radius,
            self.density,
            youngs_modulus=self.youngs_modulus,
            shear_modulus=self.shear_modulus,
        )
        self.cc_sim.append(rod_1)
        self.rods.append(rod_1)
        
        # Rod 2 (if double_rod is True)
        if self.double_rod:
            start_rod_2 = np.array([0.0, 0.0, -self.base_length])
            rod_2 = ea.CosseratRod.straight_rod(
                self.n_elem,
                start_rod_2,
                self.direction,
                self.normal,
                self.base_length,
                self.base_radius,
                self.density,
                youngs_modulus=self.youngs_modulus,
                shear_modulus=self.shear_modulus,
            )
            self.cc_sim.append(rod_2)
            self.rods.append(rod_2)
    
    def _apply_constraints_and_forces(self):
        """Apply constraints, forces, and damping to the rods."""
        
        # Rod 1 constraints and forces
        self.cc_sim.constrain(self.rods[0]).using(
            FixedConstraint,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
        )
        
        self.cc_sim.add_forcing_to(self.rods[0]).using(
            ea.UniformTorques,
            torque=0,
            direction=np.array([0, 0, 0])
        )
        
        self.cc_sim.dampen(self.rods[0]).using(
            ea.dissipation.AnalyticalLinearDamper,
            damping_constant=self.damping_constant,
            time_step=self.dt,
        )
        
        # Rod 2 forces and damping (if exists)
        if self.double_rod:
            self.cc_sim.add_forcing_to(self.rods[1]).using(
                ea.UniformTorques,
                torque=0,
                direction=np.array([0, 0, 0])
            )
            
            self.cc_sim.dampen(self.rods[1]).using(
                ea.dissipation.AnalyticalLinearDamper,
                damping_constant=self.damping_constant,
                time_step=self.dt,
            )
    
    def _connect_rods(self):
        """Connect the two rods with a fixed joint."""
        
        if len(self.rods) >= 2:
            self.cc_sim.connect(
                first_rod=self.rods[0],
                second_rod=self.rods[1],
                first_connect_idx=-1,  # Connect to the last node of the first rod
                second_connect_idx=0,  # Connect to first node of the second rod
            ).using(
                ea.FixedJoint,  # Type of connection between rods
                k=1e5,     # Spring constant of force holding rods together
                nu=0,      # Energy dissipation of joint
                kt=1,      # Rotational stiffness of rod to avoid rods twisting
                nut=0,     # Rotational damping of joint
                rest_rotation_matrix=get_relative_rotation_two_systems(
                    self.rods[0], -1, self.rods[1], 0
                ),
            )
    
    def _setup_callbacks(self):
        """Setup data collection callbacks for each rod."""
        
        for i, rod in enumerate(self.rods):
            callback_params = ea.defaultdict(list)
            self.cc_sim.collect_diagnostics(rod).using(
                DataCallback,
                step_skip=self.step_skip,
                callback_params=callback_params
            )
            self.callback_params.append(callback_params)
            
        # If targets are defined, setup their callbacks
        if hasattr(self, 'targets') and len(self.targets) > 0:
            for target in self.targets:
                target_callback_params = ea.defaultdict(list)
                self.cc_sim.collect_diagnostics(target).using(
                    RigidSphereCallBack,
                    step_skip=self.step_skip,
                    callback_params=target_callback_params
                )
                self.target_callback_params.append(target_callback_params)

    def _extract_torque_objects(self):
        """Extract torque objects for dynamic control during simulation."""
        
        self.torque_objects = []
        
        # The structure is: _ext_forces_torques contains tuples of (system_index, torque_object)
        for system_idx, torque_obj in self.cc_sim._ext_forces_torques:
            self.torque_objects.append(torque_obj)
            print(f"Found Rod {system_idx + 1} torque object: {torque_obj.torque}")
    
    def _setup_timestepper(self):
        """Setup the timestepper for the simulation."""
        
        self.timestepper = ea.PositionVerlet()
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.timestepper, self.cc_sim
        )
    
    def get_rods(self):
        """Return the list of rods in the simulation."""
        return self.rods
    
    def get_targets(self):
        """Return the list of target spheres in the simulation."""
        return self.targets
    
    def get_target_callback_params(self):
        """Return the callback parameters for target data collection."""
        return self.target_callback_params
    
    def get_torque_objects(self):
        """Return the list of torque objects for dynamic control."""
        return self.torque_objects
    
    def get_callback_params(self):
        """Return the callback parameters for data collection."""
        return self.callback_params
    
    def get_tip_state(self):
        """
        Get the state of the tip of the last rod.
        
        Returns:
        --------
        np.ndarray
            Position and orientation of the tip of the last rod
        """
        if self.rods:
            rod = self.rods[-1]
            return np.concatenate((rod.position_collection[-1], rod.director_collection[-1]))
        else:
            raise ValueError("No rods available in the simulation.")
    
    def get_target_position(self, target_idx=0):
        """
        Get the position of a target sphere.
        
        Parameters:
        -----------
        target_idx : int
            Index of the target (default: 0 for first target)
            
        Returns:
        --------
        np.ndarray
            3D position of the target center
        """
        if target_idx < len(self.targets):
            target = self.targets[target_idx]
            if hasattr(target, 'position_collection'):
                if target.position_collection.ndim == 1:
                    return target.position_collection.copy()
                else:
                    return target.position_collection[:, 0].copy()
            else:
                raise ValueError(f"Target {target_idx} does not have position information.")
        else:
            raise ValueError(f"Target index {target_idx} out of range. Available targets: {len(self.targets)}")
    
    def set_target_position(self, target_idx, new_position):
        """
        Set the position of a target sphere (for dynamic targets).
        
        Parameters:
        -----------
        target_idx : int
            Index of the target
        new_position : np.ndarray
            New 3D position for the target center
        """
        if target_idx < len(self.targets):
            target = self.targets[target_idx]
            if hasattr(target, 'position_collection'):
                if target.position_collection.ndim == 1:
                    target.position_collection[:] = new_position
                else:
                    target.position_collection[:, 0] = new_position
            else:
                raise ValueError(f"Target {target_idx} does not have position information.")
        else:
            raise ValueError(f"Target index {target_idx} out of range. Available targets: {len(self.targets)}")

    def step(self, time):
        """
        Advance the simulation by one time step.
        
        Parameters:
        -----------
        time : float
            Current simulation time
            
        Returns:
        --------
        float
            Updated simulation time
        """
        return self.do_step(self.timestepper, self.stages_and_updates, self.cc_sim, time, self.dt)
    
    def check_stability(self):
        """
        Check if the simulation is numerically stable.
        
        Returns:
        --------
        bool
            True if stable, False if unstable
        """
        from elastica._calculus import _isnan_check
        
        for i, rod in enumerate(self.rods):
            if _isnan_check(rod.position_collection):
                print(f"System broke - Rod {i + 1}")
                return False
        return True
    
    def set_torque(self, rod_idx, torque_vector):
        """
        Dynamically set the torque for a specific rod.
        
        Parameters:
        -----------
        rod_idx : int
            Index of the rod (0 for first rod, 1 for second rod)
        torque_vector : np.ndarray
            3D torque vector to apply
        """
        if rod_idx < len(self.torque_objects):
            self.torque_objects[rod_idx].torque = torque_vector
        else:
            print(f"Warning: Rod index {rod_idx} out of range")
    
    def get_simulation_info(self):
        """
        Get basic information about the simulation setup.
        
        Returns:
        --------
        dict
            Dictionary containing simulation parameters
        """
        return {
            'n_elem': self.n_elem,
            'base_length': self.base_length,
            'base_radius': self.base_radius,
            'density': self.density,
            'youngs_modulus': self.youngs_modulus,
            'dt': self.dt,
            'mpc_dt': self.mpc_dt,
            'double_rod': self.double_rod,
            'max_torque': self.max_torque,
            'num_rods': len(self.rods),
            'total_length': self.base_length * len(self.rods)
        }