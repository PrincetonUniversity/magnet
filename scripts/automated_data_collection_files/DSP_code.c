//##########################
//
// Code for MAC_DPP_V3
//
//##########################
#include "F28x_Project.h"
#include <Interrupt.h>
#include <math.h>

void InitEPwmGpio(void);
void ConfigureADC(void);
void ConfigureEPWM(void);
void InitEPwm1(void); // Synchronization
void InitEPwm3(void); // Middle switch
void InitEPwm4(void); // Upper switch
void InitEPwm5(void); // Lower switch
void EnEPwm11(void);
void DisEPwm11(void);
void InitUserGpio(void);
__interrupt void adca1_isr(void);
__interrupt void cpu_timer0_isr(void);

//#####################
void scia_echoback_init(void);
void scia_fifo_init(void);
void scia_xmit(int a);
void scia_msg(char *msg);
Uint16 LoopCount;
Uint16 ErrorCount;
//#####################

int single_test = 0;//0; // if equal to 1, then do not communicate with Python, leave at 0 (for debugging)
float DP_forced = 0.0; //0.5; // If left to 0, disabled
float DN_forced = 0.0; //0.5; // If left to 0, disabled

// For iterative operation PARAMTERS TO EDIT IN PYTHON
float Dstep = 0.1; // do not modify it so far, please
float D0min = 0.0; // 0 to include triangular waveforms, Dstep for only Trapezoidal
float D0max = 0.4; // 0 to do triangular waves only

float Fmin = 50000; // Make sure it is a multiple of Fstep
float Fmax = 510000; // Keep below 255*Fstep for proper communication
float Fpointsperdecade = 10;//20  // Make sure the same value is used in Python

MAB_PARA mab_para;

// For single open-loop operation
float freq = 500000; // in Hz
int period; // 100000000/freq -> 500=200kHz (100MHz/freq)
int carrier; // carrier. half the period 50000000/freq -> 250=200kHz

float duty_0 = 0.0; // d2 and d4 (flat parts duty cycle)
float duty_P = 0.1; // d1 (positive part)
float duty_N; // d4 (negative part)
int CMP_L; //(carrier * duty_P + 0.5)
int CMP_H; //(carrier * (1-duty_N) + 0.5)

float duty_0_now; // Variable to watch
float duty_P_now; // Variable to watch
float duty_N_now; // Variable to watch
Uint16 freq_now; // Variable to watch

int d0_int; // To sent back the correct values to Python
int dP_int;
int d0_forced_int;
int dP_forced_int;

int freqidx = 0; // Normalized frequency over the step to be sent, as the value cannot be above 255 for scia_xmit
float logF = 0.0;

int deadtime = 7;//2;//10;//;7; // in tens of ns
float freq2carrier = 50000000; // constant that gives the ratio between carrier and freq (50MHz)


int count_duty_0 = 0;
int count_duty_P = 0;
int count_freq = 0;

void main(void)
{

    //#####################
    Uint16 ReceivedChar;
    char *msg;
    //#####################

    // Initialization
    InitSysCtrl();
    InitGpio();
	InitEPwmGpio(); //Enable PWM1~PWM11
	InitUserGpio(); //Enable GPIO22~31

    //#####################
    EALLOW;
    GpioCtrlRegs.GPAMUX2.bit.GPIO28 = 1;
    GpioCtrlRegs.GPAPUD.bit.GPIO28 = 0;
    GpioCtrlRegs.GPADIR.bit.GPIO28 = 0;

    GpioCtrlRegs.GPAMUX2.bit.GPIO29 = 1;
    GpioCtrlRegs.GPAPUD.bit.GPIO29 = 0;
    GpioCtrlRegs.GPADIR.bit.GPIO29 = 1;
    EDIS;
    //#####################

    // Configure Interrupts
	DINT;
    InitPieCtrl();

    IER = 0x0000;
    IFR = 0x0000;

    InitPieVectTable();

    //#####################
    scia_fifo_init();      // Initialize the SCI FIFO
    scia_echoback_init();  // Initialize SCI for echoback

    msg = "UART is connected!\n";
    scia_msg(msg);
    //#####################

    EALLOW;
    PieVectTable.ADCA1_INT = &adca1_isr;
    PieVectTable.TIMER0_INT = &cpu_timer0_isr;
    EDIS;

    InitCpuTimers();

    // Configure ADC, EPWM, Timer
    ConfigureEPWM();
    ConfigureADC();
    ConfigCpuTimer(&CpuTimer0, 200, 10); // (*Timer, Freq, Period us)
    CpuTimer0Regs.TCR.all = 0x4001;

    // Enable Interrupts
    IER |= M_INT1; // Enable group 1 interrupts
    EINT; // Enable Global interrupt INTM
    ERTM; // Enable Global realtime interrupt DBGM

    PieCtrlRegs.PIEIER1.bit.INTx1 = 1;
    PieCtrlRegs.PIEIER1.bit.INTx7 = 1;

    GpioDataRegs.GPACLEAR.bit.GPIO24 = 1; // Set GPIO24 as zero.

    for(;;){ // Repeat indefinitely

        if (single_test==1){ //Not needed if working with Python control
            duty_N = 1 - 2 * duty_0 - duty_P;

            carrier = (int) (freq2carrier/freq + 0.5); // amplitude of the triangular carrier waveform, half the period
            period = (int) (2.0 * carrier); // period of the actual waveform, in tens of ns
            CMP_L = (int) (carrier * duty_P + 0.5);
            CMP_H = (int) (carrier * (1-duty_N) + 0.5);

            EPwm1Regs.TBPRD = carrier; // Synch (initialization)

            EPwm3Regs.TBPRD = carrier; // Middle switch (initialization)
            EPwm3Regs.CMPA.bit.CMPA = CMP_L;
            EPwm3Regs.CMPB.bit.CMPB = CMP_H;

            EPwm4Regs.TBPRD = carrier; // Upper switch (initialization)
            EPwm4Regs.CMPA.bit.CMPA = 0; // Not used
            EPwm4Regs.CMPB.bit.CMPB = CMP_L;

            EPwm5Regs.TBPRD = carrier; // Lower switch (initialization)
            EPwm5Regs.CMPA.bit.CMPA = CMP_H;
            EPwm5Regs.CMPB.bit.CMPB = 0; // Not used

            if (CMP_L >= CMP_H){
                EPwm3Regs.AQCTLA.bit.CAU = AQ_NO_ACTION;  // keep the middle switch off
                EPwm3Regs.AQCTLA.bit.CAD = AQ_NO_ACTION;
                EPwm3Regs.AQCTLA.bit.CBU = AQ_NO_ACTION;
                EPwm3Regs.AQCTLA.bit.CBD = AQ_NO_ACTION;
            }
            else{
                EPwm3Regs.AQCTLA.bit.CAU = AQ_SET;   // set with CMP_A in the rising part
                EPwm3Regs.AQCTLA.bit.CAD = AQ_CLEAR; // clear with CMP_A in the falling part
                EPwm3Regs.AQCTLA.bit.CBU = AQ_CLEAR; // clear with CMP_B in the rising part
                EPwm3Regs.AQCTLA.bit.CBD = AQ_SET;   // set with CMP_B in the falling part
            }
        }
        else{ // Let the converter run waiting for Python
            //#####################

            // Wait for inc character
//            SciaRegs.SCIFFRX.bit.RXFIFORESET = 0;
//            SciaRegs.SCIFFRX.bit.RXFIFORESET = 1;
//            while(SciaRegs.SCIFFRX.bit.RXFFST !=1) { } // wait for XRDY =1 for empty state
//            // Get character
//            ReceivedChar = SciaRegs.SCIRXBUF.all;
//            // Echo character back
//            msg = "This is a message from the DSP\n";
//            scia_msg(msg);
//            //#####################

            for(;;){ // Repeat indefinitely

                for(duty_0 = D0min; duty_0 <= D0max+Dstep*0.5; duty_0 = duty_0 + Dstep){ // Duty cycle of each of the flat parts
                    d0_int = (int) (duty_0*100.0+0.5); // Damned flooring instead of rounding
                    d0_forced_int = (int) (((1-DP_forced-DN_forced)*0.5)*100.0+0.5);
                    if (DP_forced != 0.0 || DN_forced != 0.0){
                        if (d0_int != d0_forced_int){
                            continue;
                        }
                    }
                    //#####################
                    SciaRegs.SCIFFRX.bit.RXFIFORESET = 0;
                    SciaRegs.SCIFFRX.bit.RXFIFORESET = 1;
                    while(SciaRegs.SCIFFRX.bit.RXFFST !=1) { } // wait for XRDY =1 for empty state
                    // Get character
                    ReceivedChar = SciaRegs.SCIRXBUF.all;
                    SciaRegs.SCIFFRX.bit.RXFIFORESET = 0;
                    SciaRegs.SCIFFRX.bit.RXFIFORESET = 1;

                    count_duty_0 = count_duty_0 + 1;

                    // Echo character back
                    scia_xmit(d0_int); // I think the variables are floored instead of rounded, and that may create issues with the hexstring info
                    //#####################
                    for(duty_P = Dstep; duty_P <= 1 - 2 * duty_0 - Dstep + Dstep*0.5 ; duty_P = duty_P + Dstep){ // duty of the positive voltage
                        dP_int = (int) (duty_P*100.0+0.5);
                        dP_forced_int = (int) (DP_forced*100.0+0.5);
                        if (DP_forced != 0.0 || DN_forced != 0.0){
                            if (dP_int != dP_forced_int){
                                continue;
                            }
                        }

                        duty_N = 1 - 2 * duty_0 - duty_P;

                        //if (duty_P <= 0.15 || duty_N <= 0.15){
                        //    continue;
                        //}

                        //#####################
                        SciaRegs.SCIFFRX.bit.RXFIFORESET = 0;
                        SciaRegs.SCIFFRX.bit.RXFIFORESET = 1;
                        while(SciaRegs.SCIFFRX.bit.RXFFST !=1) { } // wait for XRDY =1 for empty state
                        // Get character
                        ReceivedChar = SciaRegs.SCIRXBUF.all;
                        SciaRegs.SCIFFRX.bit.RXFIFORESET = 0;
                        SciaRegs.SCIFFRX.bit.RXFIFORESET = 1;

                        count_duty_P = count_duty_P + 1;

                        // Echo character back
                        scia_xmit(dP_int);//scia_xmit(duty_P*100);
                        //#####################

                        // Frequency loop
                        for (freqidx = 3*Fpointsperdecade; freqidx <= 7*Fpointsperdecade; freqidx++){  // From 1k to 10M --> from 3 to 7
                            logF = (float) (freqidx*1/Fpointsperdecade);
                            freq = (float) (pow(10, logF));
                            if (freq<Fmin || freq>Fmax){
                                continue;
                            }

                            //#####################
                            SciaRegs.SCIFFRX.bit.RXFIFORESET = 0;
                            SciaRegs.SCIFFRX.bit.RXFIFORESET = 1;
                            // Wait for inc character
                            while(SciaRegs.SCIFFRX.bit.RXFFST !=1) { } // wait for XRDY =1 for empty state
                            // Get character
                            ReceivedChar = SciaRegs.SCIRXBUF.all;
                            SciaRegs.SCIFFRX.bit.RXFIFORESET = 0;
                            SciaRegs.SCIFFRX.bit.RXFIFORESET = 1;

                            count_freq = count_freq + 1;

                            // Echo character back
                            scia_xmit(freqidx);
                            //#####################

                            carrier = (int) (freq2carrier/freq + 0.5); // amplitude of the triangular carrier waveform, half the period
                            period = (int) (2.0 * carrier); // period of the actual waveform, in tens of ns
                            CMP_L = (int) (carrier * duty_P + 0.5);
                            CMP_H = (int) (carrier * (1-duty_N) + 0.5);

                            EPwm1Regs.TBPRD = carrier; // Synch (initialization)

                            EPwm3Regs.TBPRD = carrier; // Middle switch (initialization)
                            EPwm3Regs.CMPA.bit.CMPA = CMP_L;
                            EPwm3Regs.CMPB.bit.CMPB = CMP_H;

                            EPwm4Regs.TBPRD = carrier; // Upper switch (initialization)
                            EPwm4Regs.CMPA.bit.CMPA = 0; // Not used
                            EPwm4Regs.CMPB.bit.CMPB = CMP_L;

                            EPwm5Regs.TBPRD = carrier; // Lower switch (initialization)
                            EPwm5Regs.CMPA.bit.CMPA = CMP_H;
                            EPwm5Regs.CMPB.bit.CMPB = 0; // Not used

                            if (CMP_L >= CMP_H){
                                EPwm3Regs.AQCTLA.bit.CAU = AQ_NO_ACTION;  // keep the middle switch off
                                EPwm3Regs.AQCTLA.bit.CAD = AQ_NO_ACTION;
                                EPwm3Regs.AQCTLA.bit.CBU = AQ_NO_ACTION;
                                EPwm3Regs.AQCTLA.bit.CBD = AQ_NO_ACTION;
                            }
                            else{
                                EPwm3Regs.AQCTLA.bit.CAU = AQ_SET;   // set with CMP_A in the rising part
                                EPwm3Regs.AQCTLA.bit.CAD = AQ_CLEAR; // clear with CMP_A in the falling part
                                EPwm3Regs.AQCTLA.bit.CBU = AQ_CLEAR; // clear with CMP_B in the rising part
                                EPwm3Regs.AQCTLA.bit.CBD = AQ_SET;   // set with CMP_B in the falling part
                            }

                            duty_0_now = duty_0; // Variable to watch
                            duty_P_now = duty_P; // Variable to watch
                            duty_N_now = duty_N; // Variable to watch
                            freq_now   = freq; // Variable to watch

                        } // Freq loop
                    } // Duty P loop
                } // Duty0 loop
            } //Infinite loop
        } // Enable communication with Python
        asm ("          NOP");
    }
}

void ConfigureADC(void)
{
	// first power up adc
	EALLOW;
	//enable ADC clock; this has been done in void InitSysCtrl(void);

	// set ADC clock;
	AdcaRegs.ADCCTL2.bit.PRESCALE = 6; //set ADCCLK divider to /1, 200MHz;
	AdcbRegs.ADCCTL2.bit.PRESCALE = 6; //set ADCCLK divider to /1

	// set signal mode and conversion resolution
	AdcSetMode(ADC_ADCA, ADC_RESOLUTION_12BIT, ADC_SIGNALMODE_SINGLE);
	AdcSetMode(ADC_ADCB, ADC_RESOLUTION_12BIT, ADC_SIGNALMODE_SINGLE);

	// power up the ADCs
	AdcaRegs.ADCCTL1.bit.ADCPWDNZ = 1;
	AdcbRegs.ADCCTL1.bit.ADCPWDNZ = 1;

	// delay for 1ms to allow ADC time to power up
	DELAY_US(1000);

	EDIS;

	EALLOW;

	Uint16 acqps;
	acqps = 24; // 125ns make sure this doesn't exceed the switching period

	// ADC Channel Select
	AdcaRegs.ADCSOC0CTL.bit.CHSEL = 0;
	AdcaRegs.ADCSOC1CTL.bit.CHSEL = 1;
	AdcaRegs.ADCSOC2CTL.bit.CHSEL = 2;
	AdcaRegs.ADCSOC3CTL.bit.CHSEL = 3;
	AdcaRegs.ADCSOC4CTL.bit.CHSEL = 4;
	AdcaRegs.ADCSOC5CTL.bit.CHSEL = 5;

	AdcbRegs.ADCSOC0CTL.bit.CHSEL = 0;
	AdcbRegs.ADCSOC1CTL.bit.CHSEL = 1;
	AdcbRegs.ADCSOC2CTL.bit.CHSEL = 2;
	AdcbRegs.ADCSOC3CTL.bit.CHSEL = 3;
	AdcbRegs.ADCSOC4CTL.bit.CHSEL = 4;
	AdcbRegs.ADCSOC5CTL.bit.CHSEL = 5;

	AdcaRegs.ADCSOC0CTL.bit.ACQPS = acqps; // ADC-A; Configure duration of sampling time
	AdcaRegs.ADCSOC1CTL.bit.ACQPS = acqps;
	AdcaRegs.ADCSOC2CTL.bit.ACQPS = acqps;
	AdcaRegs.ADCSOC3CTL.bit.ACQPS = acqps;
	AdcaRegs.ADCSOC4CTL.bit.ACQPS = acqps;
	AdcaRegs.ADCSOC5CTL.bit.ACQPS = acqps + 2;

	AdcbRegs.ADCSOC0CTL.bit.ACQPS = acqps; // ADC-B;
	AdcbRegs.ADCSOC1CTL.bit.ACQPS = acqps;
	AdcbRegs.ADCSOC2CTL.bit.ACQPS = acqps;
	AdcbRegs.ADCSOC3CTL.bit.ACQPS = acqps;
	AdcbRegs.ADCSOC4CTL.bit.ACQPS = acqps;
	AdcbRegs.ADCSOC5CTL.bit.ACQPS = acqps + 2;

	// Set ADC SOC trigger source; 6 SOCs of each module are used;
	// All use the the same trigger source to generate a sequence of conversions
	// 05h ADCTRIG5 - ePWM1, ADCSOCA will trigger the ADC

	AdcaRegs.ADCSOC0CTL.bit.TRIGSEL = 5; // ADC-A;
	AdcaRegs.ADCSOC1CTL.bit.TRIGSEL = 5;
	AdcaRegs.ADCSOC2CTL.bit.TRIGSEL = 5;
	AdcaRegs.ADCSOC3CTL.bit.TRIGSEL = 5;
	AdcaRegs.ADCSOC4CTL.bit.TRIGSEL = 5;
	AdcaRegs.ADCSOC5CTL.bit.TRIGSEL = 5;

	AdcbRegs.ADCSOC0CTL.bit.TRIGSEL = 5; // ADC-B;
	AdcbRegs.ADCSOC1CTL.bit.TRIGSEL = 5;
	AdcbRegs.ADCSOC2CTL.bit.TRIGSEL = 5;
	AdcbRegs.ADCSOC3CTL.bit.TRIGSEL = 5;
	AdcbRegs.ADCSOC4CTL.bit.TRIGSEL = 5;
	AdcbRegs.ADCSOC5CTL.bit.TRIGSEL = 5;

	AdcaRegs.ADCCTL1.bit.INTPULSEPOS  = 1; // Set ADC EOC pulse position; Set pulse positions to late
	AdcaRegs.ADCINTSEL1N2.bit.INT1SEL = 5; // Set ADC EOC source; EOC5 connect to ADCINT1
	AdcaRegs.ADCINTFLGCLR.bit.ADCINT1 = 1; // Clear interrupt flag;
	AdcaRegs.ADCINTSEL1N2.bit.INT1E   = 1; // enable ADC interrupt flag;

	EDIS;
}

void InitUserGpio(void)
{
    //EN 1 ~ EN 10: GPIO 22~31
    GPIO_SetupPinOptions(22, GPIO_OUTPUT, GPIO_OPENDRAIN);
    GPIO_SetupPinMux(22, GPIO_MUX_CPU1, 0);

    GPIO_SetupPinOptions(23, GPIO_OUTPUT, GPIO_OPENDRAIN);
    GPIO_SetupPinMux(23, GPIO_MUX_CPU1, 0);

    GPIO_SetupPinOptions(24, GPIO_OUTPUT, GPIO_OPENDRAIN);
    GPIO_SetupPinMux(24, GPIO_MUX_CPU1, 0);

    GPIO_SetupPinOptions(25, GPIO_OUTPUT, GPIO_OPENDRAIN);
    GPIO_SetupPinMux(25, GPIO_MUX_CPU1, 0);

    GPIO_SetupPinOptions(26, GPIO_OUTPUT, GPIO_OPENDRAIN);
    GPIO_SetupPinMux(26, GPIO_MUX_CPU1, 0);
}

void ConfigureEPWM(void)
{

	EALLOW;
	CpuSysRegs.PCLKCR0.bit.TBCLKSYNC        = 0;
	ClkCfgRegs.PERCLKDIVSEL.bit.EPWMCLKDIV  = 1; //EPWMCLKDIV=PLLSYSCLK/2; 100MHz; Note: For SYSCLK above 100 MHz, the EPWMCLK must be half of SYSCLK
	EDIS;

	EPwm1Regs.ETSEL.bit.SOCAEN	            = 1; // Enable SOC on A group
	EPwm1Regs.ETSEL.bit.SOCASEL	            = 1; // Select SOC when counter equal to zero
	EPwm1Regs.ETPS.bit.SOCAPRD 	            = 1; // Generate pulse on 1st event INT frequency 100kHz

	InitEPwm1();
	InitEPwm3();
	InitEPwm4();
	InitEPwm5();

	EALLOW;

	CpuSysRegs.PCLKCR0.bit.TBCLKSYNC        = 1;
	EDIS;
	EPwm1Regs.TBCTL.bit.SWFSYNC             = 1;
}

void InitEPwm1(void) // Master PWM for synch, definition of the initialization
{
	EPwm1Regs.TBCTR                 = 0;                // Time-Base Counter Register	no shadow
	EPwm1Regs.TBPRD                 = carrier;          // Period                       shadow
	EPwm1Regs.TBPHS.bit.TBPHS       = 0;                // Set Phase register to zero   no shadow
	EPwm1Regs.TBCTL.bit.CTRMODE     = TB_COUNT_UPDOWN;  // Symmetrical mode             no shadow
	EPwm1Regs.TBCTL.bit.PHSEN       = TB_DISABLE;       // Master module                no shadow
	EPwm1Regs.TBCTL.bit.PRDLD       = TB_SHADOW;        //                              no shadow
	EPwm1Regs.TBCTL.bit.HSPCLKDIV   = TB_DIV1;          // clk ratio to system clk = 1  no shadow
	EPwm1Regs.TBCTL.bit.CLKDIV      = TB_DIV1;          // clk ratio to system clk = 1  no shadow
	EPwm1Regs.TBCTL.bit.SYNCOSEL    = TB_CTR_ZERO;      // sync flow-through            no shadow
	EPwm1Regs.CMPCTL.bit.SHDWAMODE  = CC_SHADOW;        //                              no shadow
	EPwm1Regs.CMPCTL.bit.SHDWBMODE  = CC_SHADOW;	    //                              no shadow
	EPwm1Regs.CMPCTL.bit.LOADAMODE  = CC_CTR_ZERO;      // load on CTR=Zero
	EPwm1Regs.CMPCTL.bit.LOADBMODE  = CC_CTR_ZERO;      // load on CTR=Zero

	EPwm1Regs.AQCTLA.all            = 0x120;			// CAU Set CBU Clear
	EPwm1Regs.AQCTLB.all            = 0x0000; 		    // all clear

    EPwm1Regs.AQCTLA.all            = 0x1111;           // **************open-circuit the supply

	EPwm1Regs.AQSFRC.bit.RLDCSF     = CC_CTR_PRD;
	EPwm1Regs.CMPA.bit.CMPA         = 0;
	EPwm1Regs.CMPB.bit.CMPB         = 5;                // Before CMPB_sup=100
	EPwm1Regs.DBRED.bit.DBRED       = 0; 				//                              no shadow
	EPwm1Regs.DBFED.bit.DBFED       = 0; 				//                              no shadow
	EPwm1Regs.DBCTL.bit.OUT_MODE    = DB_FULL_ENABLE;   // Enable Dead-band module		no shadow
	EPwm1Regs.DBCTL.bit.IN_MODE     = DBA_ALL;	        // CMPA input					no shadow
	EPwm1Regs.DBCTL.bit.POLSEL      = DB_ACTV_HIC;      // Active Hi complementary		no shadow
}

void InitEPwm3(void) // Slave PWM for the middle switch
{
	EPwm3Regs.TBCTR                 = 0;			    // Time-Base Counter Register (carrier: lowest value)
	EPwm3Regs.TBPRD                 = carrier;          // Period = 179+1 TBCLK counts (carrier: highest value)
	EPwm3Regs.TBPHS.bit.TBPHS       = 2;                // Set Phase register to 2 to sync delay with respect to PWM1
	EPwm3Regs.TBCTL.bit.CTRMODE     = TB_COUNT_UPDOWN;  // Symmetrical mode
	EPwm3Regs.TBCTL.bit.PHSEN       = TB_ENABLE;        // Slave module
	EPwm3Regs.TBCTL.bit.PRDLD       = TB_SHADOW;
	EPwm3Regs.TBCTL.bit.HSPCLKDIV   = TB_DIV1;          // clk ratio to system clk = 1
	EPwm3Regs.TBCTL.bit.CLKDIV      = TB_DIV1;          // clk ratio to system clk = 1
	EPwm3Regs.TBCTL.bit.SYNCOSEL    = TB_SYNC_IN;       // sync flow-through
	EPwm3Regs.CMPCTL.bit.SHDWAMODE  = CC_SHADOW;
	EPwm3Regs.CMPCTL.bit.SHDWBMODE  = CC_SHADOW;
	EPwm3Regs.CMPCTL.bit.LOADAMODE  = CC_CTR_ZERO;      // load on CTR=Zero
	EPwm3Regs.CMPCTL.bit.LOADBMODE  = CC_CTR_ZERO;      // load on CTR=Zero
	//AQ_TOGGLE -> Toggle on and off; AQ_SET -> turn on; AQ_CLEAR -> Turn off; AQ_NO_ACTION -> Do nothing
	EPwm3Regs.AQCTLA.bit.CAU        = AQ_SET;           // set with CMP_A in the rising part
	EPwm3Regs.AQCTLA.bit.CAD        = AQ_CLEAR;         // clear with CMP_A in the falling part
	EPwm3Regs.AQCTLA.bit.CBU        = AQ_CLEAR;         // clear with CMP_B in the rising part
	EPwm3Regs.AQCTLA.bit.CBD        = AQ_SET;           // set with CMP_B in the falling part
	EPwm3Regs.AQCTLA.bit.ZRO        = AQ_CLEAR;         // clear with 0 just in case

	EPwm3Regs.AQCTLB.all            = 0x0000;		    // All clear (no output in B)

	EPwm3Regs.AQSFRC.bit.RLDCSF     = CC_CTR_PRD; 	    //                              no shadow
	EPwm3Regs.CMPA.bit.CMPA         = CMP_L;
	EPwm3Regs.CMPB.bit.CMPB         = CMP_H;
	EPwm3Regs.DBRED.bit.DBRED       = deadtime;         // FED = 50 TBCLKs
	EPwm3Regs.DBFED.bit.DBFED       = deadtime;         // RED = 50 TBCLKs 				no shadow
	EPwm3Regs.DBCTL.bit.OUT_MODE    = DB_FULL_ENABLE;   //                              no shadow
	EPwm3Regs.DBCTL.bit.IN_MODE     = DBA_ALL;	        // CMPA input				    no shadow
	EPwm3Regs.DBCTL.bit.POLSEL      = DB_ACTV_HIC;      // Active Hi complementary		no shadow
}

void InitEPwm4(void) // Slave PWM for the upper switch
{
	EPwm4Regs.TBCTR                 = 0;				// Time-Base Counter Register
	EPwm4Regs.TBPRD                 = carrier;          // Period = 179+1 TBCLK counts
	EPwm4Regs.TBPHS.bit.TBPHS       = 2;                // Set Phase register to 2 sync delay
	EPwm4Regs.TBCTL.bit.CTRMODE     = TB_COUNT_UPDOWN;  // Symmetrical mode
	EPwm4Regs.TBCTL.bit.PHSEN       = TB_ENABLE;        // Slave module
	EPwm4Regs.TBCTL.bit.PRDLD       = TB_SHADOW;
	EPwm4Regs.TBCTL.bit.HSPCLKDIV   = TB_DIV1;
	EPwm4Regs.TBCTL.bit.CLKDIV      = TB_DIV1;

	EPwm4Regs.TBCTL.bit.SYNCOSEL    = TB_SYNC_IN;
	EPwm4Regs.CMPCTL.bit.SHDWAMODE  = CC_SHADOW;
	EPwm4Regs.CMPCTL.bit.SHDWBMODE  = CC_SHADOW;
	EPwm4Regs.CMPCTL.bit.LOADAMODE  = CC_CTR_ZERO;      // load on CTR=Zero CTR=PRD
	EPwm4Regs.CMPCTL.bit.LOADBMODE  = CC_CTR_ZERO;      // load on CTR=Zero CTR=PRD

    EPwm4Regs.AQCTLA.bit.CAU        = AQ_NO_ACTION;     // don't listen to CMP_A
    EPwm4Regs.AQCTLA.bit.CAD        = AQ_NO_ACTION;     // don't listen to CMP_A
    EPwm4Regs.AQCTLA.bit.CBU        = AQ_CLEAR;         // clear with CMP_B in the rising part
    EPwm4Regs.AQCTLA.bit.CBD        = AQ_SET;           // set with CMP_B in the falling part
    EPwm4Regs.AQCTLA.bit.ZRO        = AQ_SET;           // set with 0 just in case

	EPwm4Regs.AQCTLB.all            = 0x0000; 		    // all clear

	EPwm4Regs.AQSFRC.bit.RLDCSF     = CC_CTR_PRD;		// no shadow
	EPwm4Regs.CMPA.bit.CMPA         = 0;                // not used
	EPwm4Regs.CMPB.bit.CMPB         = CMP_L;
	EPwm4Regs.DBRED.bit.DBRED       = deadtime;         // FED = 50 TBCLKs
	EPwm4Regs.DBFED.bit.DBFED       = deadtime;         // RED = 50 TBCLKs				no shadow
	EPwm4Regs.DBCTL.bit.OUT_MODE    = DB_FULL_ENABLE;   // no shadow
	EPwm4Regs.DBCTL.bit.IN_MODE     = DBA_ALL;	        // CMPA input					no shadow
	EPwm4Regs.DBCTL.bit.POLSEL      = DB_ACTV_HIC;      // Active Hi complementary		no shadow
}

void InitEPwm5(void) // Slave PWM for the lower switch
{
	EPwm5Regs.TBCTR                 = 0;                // Time-Base Counter Register
	EPwm5Regs.TBPRD                 = carrier;          // Period = 179+1 TBCLK counts
	EPwm5Regs.TBPHS.bit.TBPHS       = 2;                // Set Phase register to 2
	EPwm5Regs.TBCTL.bit.CTRMODE     = TB_COUNT_UPDOWN;  // Symmetrical mode
	EPwm5Regs.TBCTL.bit.PHSEN       = TB_ENABLE;        // Slave module
	EPwm5Regs.TBCTL.bit.PRDLD       = TB_SHADOW;
	EPwm5Regs.TBCTL.bit.HSPCLKDIV   = TB_DIV1;
	EPwm5Regs.TBCTL.bit.CLKDIV      = TB_DIV1;

	EPwm5Regs.TBCTL.bit.SYNCOSEL    = TB_SYNC_IN;       // sync flow-through
	EPwm5Regs.CMPCTL.bit.SHDWAMODE  = CC_SHADOW;
	EPwm5Regs.CMPCTL.bit.SHDWBMODE  = CC_SHADOW;
	EPwm5Regs.CMPCTL.bit.LOADAMODE  = CC_CTR_ZERO;      // load on CTR=Zero
	EPwm5Regs.CMPCTL.bit.LOADBMODE  = CC_CTR_ZERO;      // load on CTR=Zero

    EPwm5Regs.AQCTLA.bit.CAU        = AQ_SET;           // set with CMP_A in the rising part
    EPwm5Regs.AQCTLA.bit.CAD        = AQ_CLEAR;         // clear with CMP_A in the falling part
    EPwm5Regs.AQCTLA.bit.CBU        = AQ_NO_ACTION;     // don't listen to CMP_B
    EPwm5Regs.AQCTLA.bit.CBD        = AQ_NO_ACTION;     // don't listen to CMP_B
    EPwm5Regs.AQCTLA.bit.ZRO        = AQ_CLEAR;         // clear with 0 just in case

	EPwm5Regs.AQCTLB.all            = 0x0000;		    // all clear

	EPwm5Regs.AQSFRC.bit.RLDCSF     = CC_CTR_PRD;		//                              no shadow
	EPwm5Regs.CMPA.bit.CMPA         = CMP_H;
	EPwm5Regs.CMPB.bit.CMPB         = 0;                // Not used
	EPwm5Regs.DBRED.bit.DBRED       = deadtime;         // FED = 50 TBCLKs
	EPwm5Regs.DBFED.bit.DBFED       = deadtime;         // RED = 50 TBCLKs				no shadow
	EPwm5Regs.DBCTL.bit.OUT_MODE    = DB_FULL_ENABLE;   //                              no shadow
	EPwm5Regs.DBCTL.bit.IN_MODE     = DBA_ALL;	        // CMPA input					no shadow
	EPwm5Regs.DBCTL.bit.POLSEL      = DB_ACTV_HIC;      // Active Hi complementary		no shadow
}

__interrupt void adca1_isr(void)
{
	AdcaRegs.ADCINTFLGCLR.bit.ADCINT1   = 1;             // Clear ADCINT1 flag reinitialize for next SOC
	PieCtrlRegs.PIEACK.all              = PIEACK_GROUP1; // Acknowledge interrupt to PIE
}

__interrupt void cpu_timer0_isr(void)
{
	PieCtrlRegs.PIEACK.all = PIEACK_GROUP1;
}

//#####################
void scia_echoback_init()
{
    // Note: Clocks were turned on to the SCIA peripheral in the InitSysCtrl() function
    SciaRegs.SCICCR.all             = 0x0007;  // 1 stop bit,  No loopback. No parity,8 char bits. async mode, idle-line protocol
    SciaRegs.SCICTL1.all            = 0x0003;  // enable TX, RX, internal SCICLK. Disable RX ERR, SLEEP, TXWAKE
    SciaRegs.SCICTL2.all            = 0x0003;
    SciaRegs.SCICTL2.bit.TXINTENA   = 1;
    SciaRegs.SCICTL2.bit.RXBKINTENA = 1;

    SciaRegs.SCIHBAUD.all           = 0x0005;  // 4800 baud @LSPCLK = 50MHz (200 MHz SYSCLK).
    SciaRegs.SCILBAUD.all           = 0x0016;
    SciaRegs.SCICTL1.all            = 0x0023;  // Relinquish SCI from Reset
}

// Transmit a character from the SCI
void scia_xmit(int a)
{
    while (SciaRegs.SCIFFTX.bit.TXFFST != 0) {}
    SciaRegs.SCITXBUF.all = a;
}

void scia_msg(char * msg)
{
    int i;
    i = 0;
    while(msg[i] != '\0')
    {
        scia_xmit(msg[i]);
        i++;
    }
}

// Initialize the SCI FIFO
void scia_fifo_init()
{
    SciaRegs.SCIFFTX.all = 0xE040;
    SciaRegs.SCIFFRX.all = 0x2044;
    SciaRegs.SCIFFCT.all = 0x0;
}
//#####################
