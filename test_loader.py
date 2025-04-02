from src.pressure_loader import PressureLoader

if __name__ == "__main__":
    # Initialize the pressure loader
    pressure_loader = PressureLoader()

    # Start listening for pressure values
    pressure_loader.load_pressure()