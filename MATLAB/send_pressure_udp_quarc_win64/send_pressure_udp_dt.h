/*
 * send_pressure_udp_dt.h
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

#include "ext_types.h"

/* data type size table */
static uint_T rtDataTypeSizes[] = {
  sizeof(real_T),
  sizeof(real32_T),
  sizeof(int8_T),
  sizeof(uint8_T),
  sizeof(int16_T),
  sizeof(uint16_T),
  sizeof(int32_T),
  sizeof(uint32_T),
  sizeof(boolean_T),
  sizeof(fcn_call_T),
  sizeof(int_T),
  sizeof(pointer_T),
  sizeof(action_T),
  2*sizeof(uint32_T),
  sizeof(t_card)
};

/* data type name table */
static const char_T * rtDataTypeNames[] = {
  "real_T",
  "real32_T",
  "int8_T",
  "uint8_T",
  "int16_T",
  "uint16_T",
  "int32_T",
  "uint32_T",
  "boolean_T",
  "fcn_call_T",
  "int_T",
  "pointer_T",
  "action_T",
  "timer_uint32_pair_T",
  "t_card"
};

/* data type transitions for block I/O structure */
static DataTypeTransition rtBTransitions[] = {
  { (char_T *)(&send_pressure_udp_B.VectorConcatenate[0]), 0, 0, 3 }
  ,

  { (char_T *)(&send_pressure_udp_DW.HILInitialize_AIMinimums[0]), 0, 0, 204 },

  { (char_T *)(&send_pressure_udp_DW.HILInitialize_Card), 14, 0, 1 },

  { (char_T *)(&send_pressure_udp_DW.HILReadAnalog3_PWORK), 11, 0, 3 },

  { (char_T *)(&send_pressure_udp_DW.HILInitialize_ClockModes[0]), 6, 0, 57 },

  { (char_T *)(&send_pressure_udp_DW.HILInitialize_POSortedChans[0]), 7, 0, 8 }
};

/* data type transition table for block I/O structure */
static DataTypeTransitionTable rtBTransTable = {
  6U,
  rtBTransitions
};

/* data type transitions for Parameters structure */
static DataTypeTransition rtPTransitions[] = {
  { (char_T *)(&send_pressure_udp_P.UDPSend_remotePort), 6, 0, 1 },

  { (char_T *)(&send_pressure_udp_P.HILReadAnalog3_channels), 7, 0, 3 },

  { (char_T *)(&send_pressure_udp_P.HILInitialize_OOTerminate), 0, 0, 24 },

  { (char_T *)(&send_pressure_udp_P.HILInitialize_CKChannels[0]), 6, 0, 23 },

  { (char_T *)(&send_pressure_udp_P.HILInitialize_AIChannels[0]), 7, 0, 33 },

  { (char_T *)(&send_pressure_udp_P.HILInitialize_Active), 8, 0, 38 }
};

/* data type transition table for Parameters structure */
static DataTypeTransitionTable rtPTransTable = {
  6U,
  rtPTransitions
};

/* [EOF] send_pressure_udp_dt.h */
