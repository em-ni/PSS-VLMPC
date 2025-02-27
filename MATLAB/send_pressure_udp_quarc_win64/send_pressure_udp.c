/*
 * send_pressure_udp.c
 *
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * Code generation for model "send_pressure_udp".
 *
 * Model version              : 1.273
 * Simulink Coder version : 8.14 (R2018a) 06-Feb-2018
 * C source code generated on : Thu Feb 27 12:15:16 2025
 *
 * Target selection: quarc_win64.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: 32-bit Generic
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#include "send_pressure_udp.h"
#include "send_pressure_udp_private.h"
#include "send_pressure_udp_dt.h"

/* Block signals (default storage) */
B_send_pressure_udp_T send_pressure_udp_B;

/* Block states (default storage) */
DW_send_pressure_udp_T send_pressure_udp_DW;

/* Real-time model */
RT_MODEL_send_pressure_udp_T send_pressure_udp_M_;
RT_MODEL_send_pressure_udp_T *const send_pressure_udp_M = &send_pressure_udp_M_;

/* Model output function */
void send_pressure_udp_output(void)
{
  /* local block i/o variables */
  real_T rtb_HILReadAnalog1;

  /* S-Function (hil_read_analog_block): '<Root>/HIL Read Analog3' */

  /* S-Function Block: send_pressure_udp/HIL Read Analog3 (hil_read_analog_block) */
  {
    t_error result = hil_read_analog(send_pressure_udp_DW.HILInitialize_Card,
      &send_pressure_udp_P.HILReadAnalog3_channels, 1,
      &send_pressure_udp_DW.HILReadAnalog3_Buffer);
    if (result < 0) {
      msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
        (_rt_error_message));
      rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
    }

    rtb_HILReadAnalog1 = send_pressure_udp_DW.HILReadAnalog3_Buffer;
  }

  /* Gain: '<Root>/Gain2' incorporates:
   *  Constant: '<Root>/Constant2'
   *  Sum: '<Root>/Subtract2'
   */
  send_pressure_udp_B.VectorConcatenate[0] = (rtb_HILReadAnalog1 -
    send_pressure_udp_P.Constant2_Value) * send_pressure_udp_P.Gain2_Gain;

  /* S-Function (hil_read_analog_block): '<Root>/HIL Read Analog2' */

  /* S-Function Block: send_pressure_udp/HIL Read Analog2 (hil_read_analog_block) */
  {
    t_error result = hil_read_analog(send_pressure_udp_DW.HILInitialize_Card,
      &send_pressure_udp_P.HILReadAnalog2_channels, 1,
      &send_pressure_udp_DW.HILReadAnalog2_Buffer);
    if (result < 0) {
      msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
        (_rt_error_message));
      rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
    }

    rtb_HILReadAnalog1 = send_pressure_udp_DW.HILReadAnalog2_Buffer;
  }

  /* Gain: '<Root>/Gain7' incorporates:
   *  Constant: '<Root>/Constant'
   *  Sum: '<Root>/Subtract'
   */
  send_pressure_udp_B.VectorConcatenate[1] = (rtb_HILReadAnalog1 -
    send_pressure_udp_P.Constant_Value) * send_pressure_udp_P.Gain7_Gain;

  /* S-Function (hil_read_analog_block): '<Root>/HIL Read Analog1' */

  /* S-Function Block: send_pressure_udp/HIL Read Analog1 (hil_read_analog_block) */
  {
    t_error result = hil_read_analog(send_pressure_udp_DW.HILInitialize_Card,
      &send_pressure_udp_P.HILReadAnalog1_channels, 1,
      &send_pressure_udp_DW.HILReadAnalog1_Buffer);
    if (result < 0) {
      msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
        (_rt_error_message));
      rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
    }

    rtb_HILReadAnalog1 = send_pressure_udp_DW.HILReadAnalog1_Buffer;
  }

  /* Gain: '<Root>/Gain1' incorporates:
   *  Constant: '<Root>/Constant1'
   *  Sum: '<Root>/Subtract1'
   */
  send_pressure_udp_B.VectorConcatenate[2] = (rtb_HILReadAnalog1 -
    send_pressure_udp_P.Constant1_Value) * send_pressure_udp_P.Gain1_Gain;
}

/* Model update function */
void send_pressure_udp_update(void)
{
  char_T *sErr;

  /* Update for S-Function (sdspToNetwork): '<Root>/UDP Send' */
  sErr = GetErrorBuffer(&send_pressure_udp_DW.UDPSend_NetworkLib[0U]);
  LibUpdate_Network(&send_pressure_udp_DW.UDPSend_NetworkLib[0U],
                    &send_pressure_udp_B.VectorConcatenate[0U], 3);
  if (*sErr != 0) {
    rtmSetErrorStatus(send_pressure_udp_M, sErr);
    rtmSetStopRequested(send_pressure_udp_M, 1);
  }

  /* End of Update for S-Function (sdspToNetwork): '<Root>/UDP Send' */

  /* Update absolute time for base rate */
  /* The "clockTick0" counts the number of times the code of this task has
   * been executed. The absolute time is the multiplication of "clockTick0"
   * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
   * overflow during the application lifespan selected.
   * Timer of this task consists of two 32 bit unsigned integers.
   * The two integers represent the low bits Timing.clockTick0 and the high bits
   * Timing.clockTickH0. When the low bit overflows to 0, the high bits increment.
   */
  if (!(++send_pressure_udp_M->Timing.clockTick0)) {
    ++send_pressure_udp_M->Timing.clockTickH0;
  }

  send_pressure_udp_M->Timing.t[0] = send_pressure_udp_M->Timing.clockTick0 *
    send_pressure_udp_M->Timing.stepSize0 +
    send_pressure_udp_M->Timing.clockTickH0 *
    send_pressure_udp_M->Timing.stepSize0 * 4294967296.0;
}

/* Model initialize function */
void send_pressure_udp_initialize(void)
{
  {
    char_T *sErr;

    /* Start for S-Function (hil_initialize_block): '<Root>/HIL Initialize' */

    /* S-Function Block: send_pressure_udp/HIL Initialize (hil_initialize_block) */
    {
      t_int result;
      t_boolean is_switching;
      result = hil_open("qpid_e", "0", &send_pressure_udp_DW.HILInitialize_Card);
      if (result < 0) {
        msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
          (_rt_error_message));
        rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
        return;
      }

      is_switching = false;
      result = hil_set_card_specific_options
        (send_pressure_udp_DW.HILInitialize_Card,
         "enc0_dir=0;enc0_filter=1;enc0_a=1;enc0_b=1;enc0_z=0;enc0_reload=0;enc1_dir=0;enc1_filter=1;enc1_a=1;enc1_b=1;enc1_z=0;enc1_reload=0;enc2_dir=0;enc2_filter=1;enc2_a=1;enc2_b=1;enc2_z=0;enc2_reload=0;enc3_dir=0;enc3_filter=1;enc3_a=1;enc3_b=1;enc3_z=0;enc3_reload=0;enc4_dir=0;enc4_filter=1;enc4_a=1;enc4_b=1;enc4_z=0;enc4_reload=0;enc5_dir=0;enc5_filter=1;enc5_a=1;enc5_b=1;enc5_z=0;enc5_reload=0;enc6_dir=0;enc6_filter=1;enc6_a=1;enc6_b=1;enc6_z=0;enc6_reload=0;enc7_dir=0;enc7_filter=1;enc7_a=1;enc7_b=1;enc7_z=0;enc7_reload=0;ext_int_polarity=0;fuse_polarity=0;convert_polarity=0;watchdog_polarity=0;ext_int_watchdog=0;fuse_watchdog=0;trig1_watchdog=0;watchdog_to_trig1=0;watchdog_to_trig2=0;counter_to_trig0=0;trigger_adcs=0;latch_on_adc=0;pwm_immediate=0",
         759);
      if (result < 0) {
        msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
          (_rt_error_message));
        rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
        return;
      }

      result = hil_watchdog_clear(send_pressure_udp_DW.HILInitialize_Card);
      if (result < 0 && result != -QERR_HIL_WATCHDOG_CLEAR) {
        msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
          (_rt_error_message));
        rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
        return;
      }

      if ((send_pressure_udp_P.HILInitialize_AIPStart && !is_switching) ||
          (send_pressure_udp_P.HILInitialize_AIPEnter && is_switching)) {
        {
          int_T i1;
          real_T *dw_AIMinimums =
            &send_pressure_udp_DW.HILInitialize_AIMinimums[0];
          for (i1=0; i1 < 8; i1++) {
            dw_AIMinimums[i1] = (send_pressure_udp_P.HILInitialize_AILow);
          }
        }

        {
          int_T i1;
          real_T *dw_AIMaximums =
            &send_pressure_udp_DW.HILInitialize_AIMaximums[0];
          for (i1=0; i1 < 8; i1++) {
            dw_AIMaximums[i1] = send_pressure_udp_P.HILInitialize_AIHigh;
          }
        }

        result = hil_set_analog_input_ranges
          (send_pressure_udp_DW.HILInitialize_Card,
           send_pressure_udp_P.HILInitialize_AIChannels, 8U,
           &send_pressure_udp_DW.HILInitialize_AIMinimums[0],
           &send_pressure_udp_DW.HILInitialize_AIMaximums[0]);
        if (result < 0) {
          msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
            (_rt_error_message));
          rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
          return;
        }
      }

      if ((send_pressure_udp_P.HILInitialize_AOPStart && !is_switching) ||
          (send_pressure_udp_P.HILInitialize_AOPEnter && is_switching)) {
        {
          int_T i1;
          real_T *dw_AOMinimums =
            &send_pressure_udp_DW.HILInitialize_AOMinimums[0];
          for (i1=0; i1 < 8; i1++) {
            dw_AOMinimums[i1] = (send_pressure_udp_P.HILInitialize_AOLow);
          }
        }

        {
          int_T i1;
          real_T *dw_AOMaximums =
            &send_pressure_udp_DW.HILInitialize_AOMaximums[0];
          for (i1=0; i1 < 8; i1++) {
            dw_AOMaximums[i1] = send_pressure_udp_P.HILInitialize_AOHigh;
          }
        }

        result = hil_set_analog_output_ranges
          (send_pressure_udp_DW.HILInitialize_Card,
           send_pressure_udp_P.HILInitialize_AOChannels, 8U,
           &send_pressure_udp_DW.HILInitialize_AOMinimums[0],
           &send_pressure_udp_DW.HILInitialize_AOMaximums[0]);
        if (result < 0) {
          msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
            (_rt_error_message));
          rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
          return;
        }
      }

      if ((send_pressure_udp_P.HILInitialize_AOStart && !is_switching) ||
          (send_pressure_udp_P.HILInitialize_AOEnter && is_switching)) {
        {
          int_T i1;
          real_T *dw_AOVoltages =
            &send_pressure_udp_DW.HILInitialize_AOVoltages[0];
          for (i1=0; i1 < 8; i1++) {
            dw_AOVoltages[i1] = send_pressure_udp_P.HILInitialize_AOInitial;
          }
        }

        result = hil_write_analog(send_pressure_udp_DW.HILInitialize_Card,
          send_pressure_udp_P.HILInitialize_AOChannels, 8U,
          &send_pressure_udp_DW.HILInitialize_AOVoltages[0]);
        if (result < 0) {
          msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
            (_rt_error_message));
          rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
          return;
        }
      }

      if (send_pressure_udp_P.HILInitialize_AOReset) {
        {
          int_T i1;
          real_T *dw_AOVoltages =
            &send_pressure_udp_DW.HILInitialize_AOVoltages[0];
          for (i1=0; i1 < 8; i1++) {
            dw_AOVoltages[i1] = send_pressure_udp_P.HILInitialize_AOWatchdog;
          }
        }

        result = hil_watchdog_set_analog_expiration_state
          (send_pressure_udp_DW.HILInitialize_Card,
           send_pressure_udp_P.HILInitialize_AOChannels, 8U,
           &send_pressure_udp_DW.HILInitialize_AOVoltages[0]);
        if (result < 0) {
          msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
            (_rt_error_message));
          rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
          return;
        }
      }

      if ((send_pressure_udp_P.HILInitialize_EIPStart && !is_switching) ||
          (send_pressure_udp_P.HILInitialize_EIPEnter && is_switching)) {
        {
          int_T i1;
          int32_T *dw_QuadratureModes =
            &send_pressure_udp_DW.HILInitialize_QuadratureModes[0];
          for (i1=0; i1 < 8; i1++) {
            dw_QuadratureModes[i1] =
              send_pressure_udp_P.HILInitialize_EIQuadrature;
          }
        }

        result = hil_set_encoder_quadrature_mode
          (send_pressure_udp_DW.HILInitialize_Card,
           send_pressure_udp_P.HILInitialize_EIChannels, 8U,
           (t_encoder_quadrature_mode *)
           &send_pressure_udp_DW.HILInitialize_QuadratureModes[0]);
        if (result < 0) {
          msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
            (_rt_error_message));
          rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
          return;
        }

        {
          int_T i1;
          real_T *dw_FilterFrequency =
            &send_pressure_udp_DW.HILInitialize_FilterFrequency[0];
          for (i1=0; i1 < 8; i1++) {
            dw_FilterFrequency[i1] =
              send_pressure_udp_P.HILInitialize_EIFrequency;
          }
        }

        result = hil_set_encoder_filter_frequency
          (send_pressure_udp_DW.HILInitialize_Card,
           send_pressure_udp_P.HILInitialize_EIChannels, 8U,
           &send_pressure_udp_DW.HILInitialize_FilterFrequency[0]);
        if (result < 0) {
          msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
            (_rt_error_message));
          rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
          return;
        }
      }

      if ((send_pressure_udp_P.HILInitialize_EIStart && !is_switching) ||
          (send_pressure_udp_P.HILInitialize_EIEnter && is_switching)) {
        {
          int_T i1;
          int32_T *dw_InitialEICounts =
            &send_pressure_udp_DW.HILInitialize_InitialEICounts[0];
          for (i1=0; i1 < 8; i1++) {
            dw_InitialEICounts[i1] = send_pressure_udp_P.HILInitialize_EIInitial;
          }
        }

        result = hil_set_encoder_counts(send_pressure_udp_DW.HILInitialize_Card,
          send_pressure_udp_P.HILInitialize_EIChannels, 8U,
          &send_pressure_udp_DW.HILInitialize_InitialEICounts[0]);
        if (result < 0) {
          msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
            (_rt_error_message));
          rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
          return;
        }
      }

      if ((send_pressure_udp_P.HILInitialize_POPStart && !is_switching) ||
          (send_pressure_udp_P.HILInitialize_POPEnter && is_switching)) {
        uint32_T num_duty_cycle_modes = 0;
        uint32_T num_frequency_modes = 0;

        {
          int_T i1;
          int32_T *dw_POModeValues =
            &send_pressure_udp_DW.HILInitialize_POModeValues[0];
          for (i1=0; i1 < 8; i1++) {
            dw_POModeValues[i1] = send_pressure_udp_P.HILInitialize_POModes;
          }
        }

        result = hil_set_pwm_mode(send_pressure_udp_DW.HILInitialize_Card,
          send_pressure_udp_P.HILInitialize_POChannels, 8U, (t_pwm_mode *)
          &send_pressure_udp_DW.HILInitialize_POModeValues[0]);
        if (result < 0) {
          msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
            (_rt_error_message));
          rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
          return;
        }

        {
          int_T i1;
          const uint32_T *p_HILInitialize_POChannels =
            send_pressure_udp_P.HILInitialize_POChannels;
          int32_T *dw_POModeValues =
            &send_pressure_udp_DW.HILInitialize_POModeValues[0];
          for (i1=0; i1 < 8; i1++) {
            if (dw_POModeValues[i1] == PWM_DUTY_CYCLE_MODE || dw_POModeValues[i1]
                == PWM_ONE_SHOT_MODE || dw_POModeValues[i1] == PWM_TIME_MODE ||
                dw_POModeValues[i1] == PWM_RAW_MODE) {
              send_pressure_udp_DW.HILInitialize_POSortedChans[num_duty_cycle_modes]
                = (p_HILInitialize_POChannels[i1]);
              send_pressure_udp_DW.HILInitialize_POSortedFreqs[num_duty_cycle_modes]
                = send_pressure_udp_P.HILInitialize_POFrequency;
              num_duty_cycle_modes++;
            } else {
              send_pressure_udp_DW.HILInitialize_POSortedChans[7U -
                num_frequency_modes] = (p_HILInitialize_POChannels[i1]);
              send_pressure_udp_DW.HILInitialize_POSortedFreqs[7U -
                num_frequency_modes] =
                send_pressure_udp_P.HILInitialize_POFrequency;
              num_frequency_modes++;
            }
          }
        }

        if (num_duty_cycle_modes > 0) {
          result = hil_set_pwm_frequency(send_pressure_udp_DW.HILInitialize_Card,
            &send_pressure_udp_DW.HILInitialize_POSortedChans[0],
            num_duty_cycle_modes,
            &send_pressure_udp_DW.HILInitialize_POSortedFreqs[0]);
          if (result < 0) {
            msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
              (_rt_error_message));
            rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
            return;
          }
        }

        if (num_frequency_modes > 0) {
          result = hil_set_pwm_duty_cycle
            (send_pressure_udp_DW.HILInitialize_Card,
             &send_pressure_udp_DW.HILInitialize_POSortedChans[num_duty_cycle_modes],
             num_frequency_modes,
             &send_pressure_udp_DW.HILInitialize_POSortedFreqs[num_duty_cycle_modes]);
          if (result < 0) {
            msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
              (_rt_error_message));
            rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
            return;
          }
        }

        {
          int_T i1;
          int32_T *dw_POModeValues =
            &send_pressure_udp_DW.HILInitialize_POModeValues[0];
          for (i1=0; i1 < 8; i1++) {
            dw_POModeValues[i1] =
              send_pressure_udp_P.HILInitialize_POConfiguration;
          }
        }

        {
          int_T i1;
          int32_T *dw_POAlignValues =
            &send_pressure_udp_DW.HILInitialize_POAlignValues[0];
          for (i1=0; i1 < 8; i1++) {
            dw_POAlignValues[i1] = send_pressure_udp_P.HILInitialize_POAlignment;
          }
        }

        {
          int_T i1;
          int32_T *dw_POPolarityVals =
            &send_pressure_udp_DW.HILInitialize_POPolarityVals[0];
          for (i1=0; i1 < 8; i1++) {
            dw_POPolarityVals[i1] = send_pressure_udp_P.HILInitialize_POPolarity;
          }
        }

        result = hil_set_pwm_configuration
          (send_pressure_udp_DW.HILInitialize_Card,
           send_pressure_udp_P.HILInitialize_POChannels, 8U,
           (t_pwm_configuration *)
           &send_pressure_udp_DW.HILInitialize_POModeValues[0],
           (t_pwm_alignment *)
           &send_pressure_udp_DW.HILInitialize_POAlignValues[0],
           (t_pwm_polarity *)
           &send_pressure_udp_DW.HILInitialize_POPolarityVals[0]);
        if (result < 0) {
          msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
            (_rt_error_message));
          rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
          return;
        }

        {
          int_T i1;
          real_T *dw_POSortedFreqs =
            &send_pressure_udp_DW.HILInitialize_POSortedFreqs[0];
          for (i1=0; i1 < 8; i1++) {
            dw_POSortedFreqs[i1] = send_pressure_udp_P.HILInitialize_POLeading;
          }
        }

        {
          int_T i1;
          real_T *dw_POValues = &send_pressure_udp_DW.HILInitialize_POValues[0];
          for (i1=0; i1 < 8; i1++) {
            dw_POValues[i1] = send_pressure_udp_P.HILInitialize_POTrailing;
          }
        }

        result = hil_set_pwm_deadband(send_pressure_udp_DW.HILInitialize_Card,
          send_pressure_udp_P.HILInitialize_POChannels, 8U,
          &send_pressure_udp_DW.HILInitialize_POSortedFreqs[0],
          &send_pressure_udp_DW.HILInitialize_POValues[0]);
        if (result < 0) {
          msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
            (_rt_error_message));
          rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
          return;
        }
      }

      if ((send_pressure_udp_P.HILInitialize_POStart && !is_switching) ||
          (send_pressure_udp_P.HILInitialize_POEnter && is_switching)) {
        {
          int_T i1;
          real_T *dw_POValues = &send_pressure_udp_DW.HILInitialize_POValues[0];
          for (i1=0; i1 < 8; i1++) {
            dw_POValues[i1] = send_pressure_udp_P.HILInitialize_POInitial;
          }
        }

        result = hil_write_pwm(send_pressure_udp_DW.HILInitialize_Card,
          send_pressure_udp_P.HILInitialize_POChannels, 8U,
          &send_pressure_udp_DW.HILInitialize_POValues[0]);
        if (result < 0) {
          msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
            (_rt_error_message));
          rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
          return;
        }
      }

      if (send_pressure_udp_P.HILInitialize_POReset) {
        {
          int_T i1;
          real_T *dw_POValues = &send_pressure_udp_DW.HILInitialize_POValues[0];
          for (i1=0; i1 < 8; i1++) {
            dw_POValues[i1] = send_pressure_udp_P.HILInitialize_POWatchdog;
          }
        }

        result = hil_watchdog_set_pwm_expiration_state
          (send_pressure_udp_DW.HILInitialize_Card,
           send_pressure_udp_P.HILInitialize_POChannels, 8U,
           &send_pressure_udp_DW.HILInitialize_POValues[0]);
        if (result < 0) {
          msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
            (_rt_error_message));
          rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
          return;
        }
      }
    }

    /* Start for S-Function (sdspToNetwork): '<Root>/UDP Send' */
    sErr = GetErrorBuffer(&send_pressure_udp_DW.UDPSend_NetworkLib[0U]);
    CreateUDPInterface(&send_pressure_udp_DW.UDPSend_NetworkLib[0U]);
    if (*sErr == 0) {
      LibCreate_Network(&send_pressure_udp_DW.UDPSend_NetworkLib[0U], 1,
                        "0.0.0.0", -1, "127.0.0.1",
                        send_pressure_udp_P.UDPSend_remotePort, 8192, 8, 0);
    }

    if (*sErr == 0) {
      LibStart(&send_pressure_udp_DW.UDPSend_NetworkLib[0U]);
    }

    if (*sErr != 0) {
      DestroyUDPInterface(&send_pressure_udp_DW.UDPSend_NetworkLib[0U]);
      if (*sErr != 0) {
        rtmSetErrorStatus(send_pressure_udp_M, sErr);
        rtmSetStopRequested(send_pressure_udp_M, 1);
      }
    }

    /* End of Start for S-Function (sdspToNetwork): '<Root>/UDP Send' */
  }
}

/* Model terminate function */
void send_pressure_udp_terminate(void)
{
  char_T *sErr;

  /* Terminate for S-Function (hil_initialize_block): '<Root>/HIL Initialize' */

  /* S-Function Block: send_pressure_udp/HIL Initialize (hil_initialize_block) */
  {
    t_boolean is_switching;
    t_int result;
    t_uint32 num_final_analog_outputs = 0;
    t_uint32 num_final_pwm_outputs = 0;
    hil_task_stop_all(send_pressure_udp_DW.HILInitialize_Card);
    hil_monitor_stop_all(send_pressure_udp_DW.HILInitialize_Card);
    is_switching = false;
    if ((send_pressure_udp_P.HILInitialize_AOTerminate && !is_switching) ||
        (send_pressure_udp_P.HILInitialize_AOExit && is_switching)) {
      {
        int_T i1;
        real_T *dw_AOVoltages = &send_pressure_udp_DW.HILInitialize_AOVoltages[0];
        for (i1=0; i1 < 8; i1++) {
          dw_AOVoltages[i1] = send_pressure_udp_P.HILInitialize_AOFinal;
        }
      }

      num_final_analog_outputs = 8U;
    }

    if ((send_pressure_udp_P.HILInitialize_POTerminate && !is_switching) ||
        (send_pressure_udp_P.HILInitialize_POExit && is_switching)) {
      {
        int_T i1;
        real_T *dw_POValues = &send_pressure_udp_DW.HILInitialize_POValues[0];
        for (i1=0; i1 < 8; i1++) {
          dw_POValues[i1] = send_pressure_udp_P.HILInitialize_POFinal;
        }
      }

      num_final_pwm_outputs = 8U;
    }

    if (0
        || num_final_analog_outputs > 0
        || num_final_pwm_outputs > 0
        ) {
      /* Attempt to write the final outputs atomically (due to firmware issue in old Q2-USB). Otherwise write channels individually */
      result = hil_write(send_pressure_udp_DW.HILInitialize_Card
                         , send_pressure_udp_P.HILInitialize_AOChannels,
                         num_final_analog_outputs
                         , send_pressure_udp_P.HILInitialize_POChannels,
                         num_final_pwm_outputs
                         , NULL, 0
                         , NULL, 0
                         , &send_pressure_udp_DW.HILInitialize_AOVoltages[0]
                         , &send_pressure_udp_DW.HILInitialize_POValues[0]
                         , (t_boolean *) NULL
                         , NULL
                         );
      if (result == -QERR_HIL_WRITE_NOT_SUPPORTED) {
        t_error local_result;
        result = 0;

        /* The hil_write operation is not supported by this card. Write final outputs for each channel type */
        if (num_final_analog_outputs > 0) {
          local_result = hil_write_analog
            (send_pressure_udp_DW.HILInitialize_Card,
             send_pressure_udp_P.HILInitialize_AOChannels,
             num_final_analog_outputs,
             &send_pressure_udp_DW.HILInitialize_AOVoltages[0]);
          if (local_result < 0) {
            result = local_result;
          }
        }

        if (num_final_pwm_outputs > 0) {
          local_result = hil_write_pwm(send_pressure_udp_DW.HILInitialize_Card,
            send_pressure_udp_P.HILInitialize_POChannels, num_final_pwm_outputs,
            &send_pressure_udp_DW.HILInitialize_POValues[0]);
          if (local_result < 0) {
            result = local_result;
          }
        }

        if (result < 0) {
          msg_get_error_messageA(NULL, result, _rt_error_message, sizeof
            (_rt_error_message));
          rtmSetErrorStatus(send_pressure_udp_M, _rt_error_message);
        }
      }
    }

    hil_task_delete_all(send_pressure_udp_DW.HILInitialize_Card);
    hil_monitor_delete_all(send_pressure_udp_DW.HILInitialize_Card);
    hil_close(send_pressure_udp_DW.HILInitialize_Card);
    send_pressure_udp_DW.HILInitialize_Card = NULL;
  }

  /* Terminate for S-Function (sdspToNetwork): '<Root>/UDP Send' */
  sErr = GetErrorBuffer(&send_pressure_udp_DW.UDPSend_NetworkLib[0U]);
  LibTerminate(&send_pressure_udp_DW.UDPSend_NetworkLib[0U]);
  if (*sErr != 0) {
    rtmSetErrorStatus(send_pressure_udp_M, sErr);
    rtmSetStopRequested(send_pressure_udp_M, 1);
  }

  LibDestroy(&send_pressure_udp_DW.UDPSend_NetworkLib[0U], 1);
  DestroyUDPInterface(&send_pressure_udp_DW.UDPSend_NetworkLib[0U]);

  /* End of Terminate for S-Function (sdspToNetwork): '<Root>/UDP Send' */
}

/*========================================================================*
 * Start of Classic call interface                                        *
 *========================================================================*/
void MdlOutputs(int_T tid)
{
  send_pressure_udp_output();
  UNUSED_PARAMETER(tid);
}

void MdlUpdate(int_T tid)
{
  send_pressure_udp_update();
  UNUSED_PARAMETER(tid);
}

void MdlInitializeSizes(void)
{
}

void MdlInitializeSampleTimes(void)
{
}

void MdlInitialize(void)
{
}

void MdlStart(void)
{
  send_pressure_udp_initialize();
}

void MdlTerminate(void)
{
  send_pressure_udp_terminate();
}

/* Registration function */
RT_MODEL_send_pressure_udp_T *send_pressure_udp(void)
{
  /* Registration code */

  /* initialize non-finites */
  rt_InitInfAndNaN(sizeof(real_T));

  /* initialize real-time model */
  (void) memset((void *)send_pressure_udp_M, 0,
                sizeof(RT_MODEL_send_pressure_udp_T));

  /* Initialize timing info */
  {
    int_T *mdlTsMap = send_pressure_udp_M->Timing.sampleTimeTaskIDArray;
    mdlTsMap[0] = 0;
    send_pressure_udp_M->Timing.sampleTimeTaskIDPtr = (&mdlTsMap[0]);
    send_pressure_udp_M->Timing.sampleTimes =
      (&send_pressure_udp_M->Timing.sampleTimesArray[0]);
    send_pressure_udp_M->Timing.offsetTimes =
      (&send_pressure_udp_M->Timing.offsetTimesArray[0]);

    /* task periods */
    send_pressure_udp_M->Timing.sampleTimes[0] = (0.005);

    /* task offsets */
    send_pressure_udp_M->Timing.offsetTimes[0] = (0.0);
  }

  rtmSetTPtr(send_pressure_udp_M, &send_pressure_udp_M->Timing.tArray[0]);

  {
    int_T *mdlSampleHits = send_pressure_udp_M->Timing.sampleHitArray;
    mdlSampleHits[0] = 1;
    send_pressure_udp_M->Timing.sampleHits = (&mdlSampleHits[0]);
  }

  rtmSetTFinal(send_pressure_udp_M, -1);
  send_pressure_udp_M->Timing.stepSize0 = 0.005;

  /* External mode info */
  send_pressure_udp_M->Sizes.checksums[0] = (2737048710U);
  send_pressure_udp_M->Sizes.checksums[1] = (3156944781U);
  send_pressure_udp_M->Sizes.checksums[2] = (631947801U);
  send_pressure_udp_M->Sizes.checksums[3] = (989028800U);

  {
    static const sysRanDType rtAlwaysEnabled = SUBSYS_RAN_BC_ENABLE;
    static RTWExtModeInfo rt_ExtModeInfo;
    static const sysRanDType *systemRan[1];
    send_pressure_udp_M->extModeInfo = (&rt_ExtModeInfo);
    rteiSetSubSystemActiveVectorAddresses(&rt_ExtModeInfo, systemRan);
    systemRan[0] = &rtAlwaysEnabled;
    rteiSetModelMappingInfoPtr(send_pressure_udp_M->extModeInfo,
      &send_pressure_udp_M->SpecialInfo.mappingInfo);
    rteiSetChecksumsPtr(send_pressure_udp_M->extModeInfo,
                        send_pressure_udp_M->Sizes.checksums);
    rteiSetTPtr(send_pressure_udp_M->extModeInfo, rtmGetTPtr(send_pressure_udp_M));
  }

  send_pressure_udp_M->solverInfoPtr = (&send_pressure_udp_M->solverInfo);
  send_pressure_udp_M->Timing.stepSize = (0.005);
  rtsiSetFixedStepSize(&send_pressure_udp_M->solverInfo, 0.005);
  rtsiSetSolverMode(&send_pressure_udp_M->solverInfo, SOLVER_MODE_SINGLETASKING);

  /* block I/O */
  send_pressure_udp_M->blockIO = ((void *) &send_pressure_udp_B);

  {
    send_pressure_udp_B.VectorConcatenate[0] = 0.0;
    send_pressure_udp_B.VectorConcatenate[1] = 0.0;
    send_pressure_udp_B.VectorConcatenate[2] = 0.0;
  }

  /* parameters */
  send_pressure_udp_M->defaultParam = ((real_T *)&send_pressure_udp_P);

  /* states (dwork) */
  send_pressure_udp_M->dwork = ((void *) &send_pressure_udp_DW);
  (void) memset((void *)&send_pressure_udp_DW, 0,
                sizeof(DW_send_pressure_udp_T));

  {
    int32_T i;
    for (i = 0; i < 8; i++) {
      send_pressure_udp_DW.HILInitialize_AIMinimums[i] = 0.0;
    }
  }

  {
    int32_T i;
    for (i = 0; i < 8; i++) {
      send_pressure_udp_DW.HILInitialize_AIMaximums[i] = 0.0;
    }
  }

  {
    int32_T i;
    for (i = 0; i < 8; i++) {
      send_pressure_udp_DW.HILInitialize_AOMinimums[i] = 0.0;
    }
  }

  {
    int32_T i;
    for (i = 0; i < 8; i++) {
      send_pressure_udp_DW.HILInitialize_AOMaximums[i] = 0.0;
    }
  }

  {
    int32_T i;
    for (i = 0; i < 8; i++) {
      send_pressure_udp_DW.HILInitialize_AOVoltages[i] = 0.0;
    }
  }

  {
    int32_T i;
    for (i = 0; i < 8; i++) {
      send_pressure_udp_DW.HILInitialize_FilterFrequency[i] = 0.0;
    }
  }

  {
    int32_T i;
    for (i = 0; i < 8; i++) {
      send_pressure_udp_DW.HILInitialize_POSortedFreqs[i] = 0.0;
    }
  }

  {
    int32_T i;
    for (i = 0; i < 8; i++) {
      send_pressure_udp_DW.HILInitialize_POValues[i] = 0.0;
    }
  }

  send_pressure_udp_DW.HILReadAnalog3_Buffer = 0.0;
  send_pressure_udp_DW.HILReadAnalog2_Buffer = 0.0;
  send_pressure_udp_DW.HILReadAnalog1_Buffer = 0.0;

  {
    int32_T i;
    for (i = 0; i < 137; i++) {
      send_pressure_udp_DW.UDPSend_NetworkLib[i] = 0.0;
    }
  }

  /* data type transition information */
  {
    static DataTypeTransInfo dtInfo;
    (void) memset((char_T *) &dtInfo, 0,
                  sizeof(dtInfo));
    send_pressure_udp_M->SpecialInfo.mappingInfo = (&dtInfo);
    dtInfo.numDataTypes = 15;
    dtInfo.dataTypeSizes = &rtDataTypeSizes[0];
    dtInfo.dataTypeNames = &rtDataTypeNames[0];

    /* Block I/O transition table */
    dtInfo.BTransTable = &rtBTransTable;

    /* Parameters transition table */
    dtInfo.PTransTable = &rtPTransTable;
  }

  /* Initialize Sizes */
  send_pressure_udp_M->Sizes.numContStates = (0);/* Number of continuous states */
  send_pressure_udp_M->Sizes.numY = (0);/* Number of model outputs */
  send_pressure_udp_M->Sizes.numU = (0);/* Number of model inputs */
  send_pressure_udp_M->Sizes.sysDirFeedThru = (0);/* The model is not direct feedthrough */
  send_pressure_udp_M->Sizes.numSampTimes = (1);/* Number of sample times */
  send_pressure_udp_M->Sizes.numBlocks = (15);/* Number of blocks */
  send_pressure_udp_M->Sizes.numBlockIO = (1);/* Number of block outputs */
  send_pressure_udp_M->Sizes.numBlockPrms = (122);/* Sum of parameter "widths" */
  return send_pressure_udp_M;
}

/*========================================================================*
 * End of Classic call interface                                          *
 *========================================================================*/
