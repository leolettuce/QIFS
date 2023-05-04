/*
 Authors: Nikita Gourianov
 (c) Nikita Gourianov 2019
*/

/*! \file tntMpsPde2DCreateDiffs.c
 *  \brief This file contains the functions which create differencing operators along various dimensions (e.g. 2nd order difference along y/dimension 2). *
 */

/* Include header for NS and PDE functions, along with for MPS library */
#include "tntPde.h"

/*! \ingroup pde
 *
 * TOWRITE
 */
tntNetwork tntMpsPde2DCreateOrder1Diff1(char BC[], unsigned L){
    
    if(L < 5){
        tntErrorPrint("Cannot Create 1st order differencer when length of MPS is less than 5. Exiting."); /* NO_COVERAGE */
        exit(-1);
    }
    
    /* Declare necessary variables */
    tntNetwork d1;
    tntNode Dstart, Dmid, Dending, Dend, leftTerminator, rightTerminator, Dmidc; //Needed to produce d1.
    tntComplexArray leftTerminator_array, rightTerminator_array, Dending_array, Dend_array, Dmid_array; //Arrays used to produce D nodes
    double h = pow((double) 1/2, (double) L);
    double dummy;
    
    /* Initialise the arrays to zeros */
    leftTerminator_array = tntComplexArrayAlloc(3);
    Dmid_array = tntComplexArrayAlloc(12,12); //bulk node
    Dending_array = tntComplexArrayAlloc(12,20); //second-to-last node
    Dend_array = tntComplexArrayAlloc(20,24); //last node
    rightTerminator_array = tntComplexArrayAlloc(6);
    
    /* Fill arrays with appropriate values. Indices are DL (fast multiindex) and UR. */
    /* Start with the left terminator array for Dstart */
    /* Depending on inputed boundary conditions, produce the left terminating node */
    if(!strcmp(BC, "Periodic") ){
        
        /* For periodic boundary conditions, left/right shifted wavefunctions at the boundaries just means they go to the other side of the domain. This can be done by filling the left terminator array with only 1s (as both an incoming 0 or 1 on the right leg is acceptable) */
        leftTerminator_array.vals[0].re = 1;
        leftTerminator_array.vals[1].re = 1;
        leftTerminator_array.vals[2].re = 1;
        
    }
    else if(!strcmp(BC,"Homogenous Dirichlet") ){
        
        /* For Homogenous Dirichlet boundary conditions, we need to make sure that the left/right shifted wavefunctions are zero whenever they're outside the domain. This can be done by letting the terminator start nodes take the value "0" when there's an incoming "1" on their right leg, as this means we're leaving the domain.  */
        leftTerminator_array.vals[0].re = 1;
    }
    else if(!strcmp(BC,"Periodic Reflective") ){
        
        /* In mirror BCs, the wavefunction is periodic, and at the boundary constitutes an axis/plane of symmetry. The first derivative is always zero in this case, while the 2nd derivative equals 2*valueAtBoundary - 2*valueJustRightOfBoundary */
        leftTerminator_array.vals[0].re = 1;
        leftTerminator_array.vals[1].re = -1;
        leftTerminator_array.vals[2].re = -1;
    }
    else{
        tntErrorPrint("The boundary conditions (BC) inputed into tntMpsPde2DCreateProlongator are not supported. Exiting.");
        exit(1);
    }
    
    /* Create the bulk nodes and the two last nodes */
    /* Dmid */
    /* When sending in 0 from the right: do nothing */
    Dmid_array.vals[0 + 12*0].re = 1;
    Dmid_array.vals[1 + 12*1].re = 1;
    Dmid_array.vals[2 + 12*2].re = 1;
    Dmid_array.vals[3 + 12*3].re = 1;
    
    /* When sending in 1 from the right: add */
    Dmid_array.vals[1+4*0 + 12*(0+4*1)].re = 1;
    Dmid_array.vals[0+4*1 + 12*(1+4*1)].re = 1;
    Dmid_array.vals[3+4*0 + 12*(2+4*1)].re = 1; //01->11
    Dmid_array.vals[2+4*1 + 12*(3+4*1)].re = 1; //11->01+
    
    /* When sending in 2 from the right: substract */
    Dmid_array.vals[1+4*2 + 12*(0+4*2)].re = 1;
    Dmid_array.vals[0+4*0 + 12*(1+4*2)].re = 1;
    Dmid_array.vals[3+4*2 + 12*(2+4*2)].re = 1; //01->11-
    Dmid_array.vals[2+4*0 + 12*(3+4*2)].re = 1; //11->01
    
    /* Dending *
    /* When sending in 0 from the right: do nothing */
    Dending_array.vals[0 + 12*0].re = 1;
    Dending_array.vals[1 + 12*1].re = 1;
    Dending_array.vals[2 + 12*2].re = 1;
    Dending_array.vals[3 + 12*3].re = 1;
    
    /* When sending in 1 from the right: add */
    Dending_array.vals[1+4*0 + 12*(0+4*1)].re = 1;
    Dending_array.vals[0+4*1 + 12*(1+4*1)].re = 1;
    Dending_array.vals[3+4*0 + 12*(2+4*1)].re = 1;
    Dending_array.vals[2+4*1 + 12*(3+4*1)].re = 1;
    
    /* When sending in 2 from the right: substract */
    Dending_array.vals[1+4*2 + 12*(0+4*2)].re = 1;
    Dending_array.vals[0+4*0 + 12*(1+4*2)].re = 1;
    Dending_array.vals[3+4*2 + 12*(2+4*2)].re = 1;
    Dending_array.vals[2+4*0 + 12*(3+4*2)].re = 1;
    
    /* When sending in 3 from the right: add 4h */
    Dending_array.vals[0+4*1 + 12*(0+4*3)].re = 1; //add
    Dending_array.vals[1+4*1 + 12*(1+4*3)].re = 1; //add
    Dending_array.vals[2+4*1 + 12*(2+4*3)].re = 1; //add
    Dending_array.vals[3+4*1 + 12*(3+4*3)].re = 1; //add
    
    /* When sending in 4 from the right: substract 4h */
    Dending_array.vals[0+4*2 + 12*(0+4*4)].re = 1; //substract
    Dending_array.vals[1+4*2 + 12*(1+4*4)].re = 1; //substract
    Dending_array.vals[2+4*2 + 12*(2+4*4)].re = 1; //substract
    Dending_array.vals[3+4*2 + 12*(3+4*4)].re = 1; //substract
    
    /* Dend */
    /* When sending in 0 from the right: do nothing */
    Dend_array.vals[0 + 20*0].re = 1;
    Dend_array.vals[1 + 20*1].re = 1;
    Dend_array.vals[2 + 20*2].re = 1;
    Dend_array.vals[3 + 20*3].re = 1;
    
    /* When sending in 1 from the right: add */
    Dend_array.vals[1+4*0 + 20*(0+4*1)].re = 1;
    Dend_array.vals[0+4*1 + 20*(1+4*1)].re = 1;
    Dend_array.vals[3+4*0 + 20*(2+4*1)].re = 1;
    Dend_array.vals[2+4*1 + 20*(3+4*1)].re = 1;
    
    /* When sending in 2 from the right: substract */
    Dend_array.vals[1+4*2 + 20*(0+4*2)].re = 1;
    Dend_array.vals[0+4*0 + 20*(1+4*2)].re = 1;
    Dend_array.vals[3+4*2 + 20*(2+4*2)].re = 1;
    Dend_array.vals[2+4*0 + 20*(3+4*2)].re = 1;
    
    /* When sending in 3 from the right: go to 2h */
    Dend_array.vals[0+4*1 + 20*(0+4*3)].re = -1; //add
    Dend_array.vals[0+4*2 + 20*(0+4*3)].re = +1; //substract
    Dend_array.vals[1+4*1 + 20*(1+4*3)].re = -1; //add
    Dend_array.vals[1+4*2 + 20*(1+4*3)].re = +1; //substract
    Dend_array.vals[2+4*1 + 20*(2+4*3)].re = -1; //add
    Dend_array.vals[2+4*2 + 20*(2+4*3)].re = +1; //substract
    Dend_array.vals[3+4*1 + 20*(3+4*3)].re = -1; //add
    Dend_array.vals[3+4*2 + 20*(3+4*3)].re = +1; //substract
    
    /* When sending in 4 from the right: go to 3h */
    /* Perform the addition & substraction at this end node, and send instructions to next node */
    Dend_array.vals[1+4*1 + 20*(0+4*4)].re = -1; //add
    Dend_array.vals[0+4*2 + 20*(1+4*4)].re = +1; //substract
    Dend_array.vals[3+4*1 + 20*(2+4*4)].re = -1; //add
    Dend_array.vals[2+4*2 + 20*(3+4*4)].re = +1; //substract
    /* Special case when remainder must be transferred */
    Dend_array.vals[1+4*4 + 20*(0+4*4)].re = +1; //substract 4h
    Dend_array.vals[0+4*3 + 20*(1+4*4)].re = -1; //add 4h
    Dend_array.vals[3+4*4 + 20*(2+4*4)].re = +1; //substract 4h
    Dend_array.vals[2+4*3 + 20*(3+4*4)].re = -1; //add 4h
    
    /* When sending in 5 from the right: go to 4h */
    /* Add 4h */
    Dend_array.vals[0+4*3 + 20*(0+4*5)].re = -1;
    Dend_array.vals[1+4*3 + 20*(1+4*5)].re = -1;
    Dend_array.vals[2+4*3 + 20*(2+4*5)].re = -1;
    Dend_array.vals[3+4*3 + 20*(3+4*5)].re = -1;
    /* Substract 4h */
    Dend_array.vals[0+4*4 + 20*(0+4*5)].re = +1;
    Dend_array.vals[1+4*4 + 20*(1+4*5)].re = +1;
    Dend_array.vals[2+4*4 + 20*(2+4*5)].re = +1;
    Dend_array.vals[3+4*4 + 20*(3+4*5)].re = +1;
    
    /* Right terminating array (for Dend) */
    /* 8th order */
    rightTerminator_array.vals[0].re = 0; //do nothing
    rightTerminator_array.vals[1].re = -4/(5*h);  //add at h
    rightTerminator_array.vals[2].re = +4/(5*h);  //substract at h
    rightTerminator_array.vals[3].re = -1/(5*h);  //add & substract at 2h
    rightTerminator_array.vals[4].re = +4/(105*h);//add & substract at 3h
    rightTerminator_array.vals[5].re = -1/(280*h);//add & substract at 4h
    
//    /* 2nd order */
//    rightTerminator_array.vals[0].re = 0;         //do nothing
//    rightTerminator_array.vals[1].re = -1/(2*h);  //add at h
//    rightTerminator_array.vals[2].re = +1/(2*h);  //substract at h
//    rightTerminator_array.vals[3].re = 0;         //add & substract at 2h
//    rightTerminator_array.vals[4].re = 0;         //add & substract at 3h
//    rightTerminator_array.vals[5].re = 0;         //add & substract at 4h
    
    /* Create the nodes */
    Dmid = tntNodeCreate(&Dmid_array, "DLUR", 4, 3, 4, 3);
    Dending = tntNodeCreate(&Dending_array, "DLUR", 4, 3, 4, 5);
    Dend = tntNodeCreate(&Dend_array, "DLUR", 4, 5, 4, 6);
    rightTerminator = tntNodeCreate(&rightTerminator_array, "LR", 6, 1);
    leftTerminator = tntNodeCreate(&leftTerminator_array, "LR", 1, 3);
    /* Use terminating nodes to produce starting and ending D nodes */
    /* First Dstart */
    Dmidc = tntNodeCopy(Dmid);
    tntNodeJoin(leftTerminator,"R",Dmidc,"L");
    Dstart = tntNodeContract(leftTerminator,Dmidc);
    /* Then Dend */
    tntNodeJoin(Dend,"R",rightTerminator,"L");
    Dend = tntNodeContract(rightTerminator,Dend);
    
    /* Produce the full network */
    /* First initialise it */
    d1 = tntNetworkCreate();
    
    /* Insert the bulk nodes */
    for(unsigned i = 0; i<L-3; i++){
        /* Create mid node at i */
        Dmidc = tntNodeCopy(Dmid);
        tntNodeInsertAtStart(Dmidc, "L", "R", d1);
    }
    
    /* Finally, insert the terminating nodes */
    tntNodeInsertAtStart(Dstart,"L","R",d1);
    tntNodeInsertAtEnd(Dending,"L","R",d1);
    tntNodeInsertAtEnd(Dend,"L","R",d1);

    /* Make tensors go from sparse to dense (and hence decrease the bond dimension) */
    dummy += tntMpoTruncate(d1, -1);
    
    /* Now free everything except d1, Dmidc, the start, ending and end D nodes. */
    tntNodeFree(&Dmid);
    tntComplexArrayFree(&leftTerminator_array);
    tntComplexArrayFree(&Dmid_array);
    tntComplexArrayFree(&rightTerminator_array);

    /* Return d1 */
    return d1;
}

/*! \ingroup pde
 *
 * TOWRITE
 */
tntNetwork tntMpsPde2DCreateOrder1Diff2(char BC[], unsigned L){
    
    if(L < 5){
        tntErrorPrint("Cannot Create 1st order differencer when length of MPS is less than 5. Exiting."); /* NO_COVERAGE */
        exit(-1);
    }
    
    /* Declare necessary variables */
    tntNetwork d2;
    tntNode Dstart, Dmid, Dending, Dend, leftTerminator, rightTerminator, Dmidc; //Needed to produce d2.
    tntComplexArray leftTerminator_array, rightTerminator_array, Dending_array, Dend_array, Dmid_array; //Arrays used to produce D nodes
    double h = pow((double) 1/2, (double) L);
    double dummy;
    
    /* Initialise the start and end arrays to zeros */
    leftTerminator_array = tntComplexArrayAlloc(3);
    Dmid_array = tntComplexArrayAlloc(12,12); //bulk node
    Dending_array = tntComplexArrayAlloc(12,20); //second-to-last node
    Dend_array = tntComplexArrayAlloc(20,24); //last node
    rightTerminator_array = tntComplexArrayAlloc(6);
    
    /* Fill arrays with appropriate values. Indices are DL (fast multiindex) and UR. */
    /* Start with the left terminator array for Dstart */
    /* Depending on inputed boundary conditions, produce the left terminating node */
    if(!strcmp(BC, "Periodic") ){
        
        /* For periodic boundary conditions, left/right shifted wavefunctions at the boundaries just means they go to the other side of the domain. This can be done by filling the left terminator array with only 1s (as both an incoming 0 or 1 on the right leg is acceptable) */
        leftTerminator_array.vals[0].re = 1;
        leftTerminator_array.vals[1].re = 1;
        leftTerminator_array.vals[2].re = 1;
        
    }
    else if(!strcmp(BC,"Homogenous Dirichlet") ){
        
        /* For Homogenous Dirichlet boundary conditions, we need to make sure that the left/right shifted wavefunctions are zero whenever they're outside the domain. This can be done by letting the terminator start nodes take the value "0" when there's an incoming "1" on their right leg, as this means we're leaving the domain.  */
        leftTerminator_array.vals[0].re = 1;
    }
    else if(!strcmp(BC,"Periodic Reflective") ){
        
        /* In mirror BCs, the wavefunction is periodic, and at the boundary constitutes an axis/plane of symmetry. The first derivative is always zero in this case, while the 2nd derivative equals 2*valueAtBoundary - 2*valueJustRightOfBoundary */
        leftTerminator_array.vals[0].re = 1;
        leftTerminator_array.vals[1].re = -1;
        leftTerminator_array.vals[2].re = -1;
    }
    else{
        tntErrorPrint("The boundary conditions (BC) inputed into tntMpsPde2DCreateProlongator are not supported. Exiting.");
        exit(1);
    }
    
    /* Create the bulk nodes and the two last nodes */
    /* Dmid */
    /* When sending in 0 from the right: do nothing */
    Dmid_array.vals[0 + 12*0].re = 1;
    Dmid_array.vals[1 + 12*1].re = 1;
    Dmid_array.vals[2 + 12*2].re = 1;
    Dmid_array.vals[3 + 12*3].re = 1;
    
    /* When sending in 1 from the right: add */
    Dmid_array.vals[2+4*0 + 12*(0+4*1)].re = 1;
    Dmid_array.vals[3+4*0 + 12*(1+4*1)].re = 1;
    Dmid_array.vals[0+4*1 + 12*(2+4*1)].re = 1;
    Dmid_array.vals[1+4*1 + 12*(3+4*1)].re = 1;
    
    /* When sending in 2 from the right: substract */
    Dmid_array.vals[2+4*2 + 12*(0+4*2)].re = 1;
    Dmid_array.vals[3+4*2 + 12*(1+4*2)].re = 1;
    Dmid_array.vals[0+4*0 + 12*(2+4*2)].re = 1;
    Dmid_array.vals[1+4*0 + 12*(3+4*2)].re = 1;
    
    /* Dending *
    /* When sending in 0 from the right: do nothing */
    Dending_array.vals[0 + 12*0].re = 1;
    Dending_array.vals[1 + 12*1].re = 1;
    Dending_array.vals[2 + 12*2].re = 1;
    Dending_array.vals[3 + 12*3].re = 1;
    
    /* When sending in 1 from the right: add */
    Dending_array.vals[2+4*0 + 12*(0+4*1)].re = 1;
    Dending_array.vals[3+4*0 + 12*(1+4*1)].re = 1;
    Dending_array.vals[0+4*1 + 12*(2+4*1)].re = 1;
    Dending_array.vals[1+4*1 + 12*(3+4*1)].re = 1;
    
    /* When sending in 2 from the right: substract */
    Dending_array.vals[2+4*2 + 12*(0+4*2)].re = 1;
    Dending_array.vals[3+4*2 + 12*(1+4*2)].re = 1;
    Dending_array.vals[0+4*0 + 12*(2+4*2)].re = 1;
    Dending_array.vals[1+4*0 + 12*(3+4*2)].re = 1;
    
    /* When sending in 3 from the right: add 4h */
    Dending_array.vals[0+4*1 + 12*(0+4*3)].re = 1; //add
    Dending_array.vals[1+4*1 + 12*(1+4*3)].re = 1; //add
    Dending_array.vals[2+4*1 + 12*(2+4*3)].re = 1; //add
    Dending_array.vals[3+4*1 + 12*(3+4*3)].re = 1; //add
    
    /* When sending in 4 from the right: substract 4h */
    Dending_array.vals[0+4*2 + 12*(0+4*4)].re = 1; //substract
    Dending_array.vals[1+4*2 + 12*(1+4*4)].re = 1; //substract
    Dending_array.vals[2+4*2 + 12*(2+4*4)].re = 1; //substract
    Dending_array.vals[3+4*2 + 12*(3+4*4)].re = 1; //substract
    
    /* Dend */
    /* When sending in 0 from the right: do nothing */
    Dend_array.vals[0 + 20*0].re = 1;
    Dend_array.vals[1 + 20*1].re = 1;
    Dend_array.vals[2 + 20*2].re = 1;
    Dend_array.vals[3 + 20*3].re = 1;
    
    /* When sending in 1 from the right: add */
    Dend_array.vals[2+4*0 + 20*(0+4*1)].re = 1;
    Dend_array.vals[3+4*0 + 20*(1+4*1)].re = 1;
    Dend_array.vals[0+4*1 + 20*(2+4*1)].re = 1;
    Dend_array.vals[1+4*1 + 20*(3+4*1)].re = 1;
    
    /* When sending in 2 from the right: substract */
    Dend_array.vals[2+4*2 + 20*(0+4*2)].re = 1;
    Dend_array.vals[3+4*2 + 20*(1+4*2)].re = 1;
    Dend_array.vals[0+4*0 + 20*(2+4*2)].re = 1;
    Dend_array.vals[1+4*0 + 20*(3+4*2)].re = 1;
    
    /* When sending in 3 from the right: go to 2h */
    Dend_array.vals[0+4*1 + 20*(0+4*3)].re = -1; //add
    Dend_array.vals[0+4*2 + 20*(0+4*3)].re = +1; //substract
    Dend_array.vals[1+4*1 + 20*(1+4*3)].re = -1; //add
    Dend_array.vals[1+4*2 + 20*(1+4*3)].re = +1; //substract
    Dend_array.vals[2+4*1 + 20*(2+4*3)].re = -1; //add
    Dend_array.vals[2+4*2 + 20*(2+4*3)].re = +1; //substract
    Dend_array.vals[3+4*1 + 20*(3+4*3)].re = -1; //add
    Dend_array.vals[3+4*2 + 20*(3+4*3)].re = +1; //substract
    
    /* When sending in 4 from the right: go to 3h */
    /* Perform the addition & substraction at this end node, and send instructions to next node */
    Dend_array.vals[2+4*1 + 20*(0+4*4)].re = -1; //add
    Dend_array.vals[3+4*1 + 20*(1+4*4)].re = -1; //add
    Dend_array.vals[0+4*2 + 20*(2+4*4)].re = +1; //substract
    Dend_array.vals[1+4*2 + 20*(3+4*4)].re = +1; //substract
    /* Special case when remainder must be transferred */
    Dend_array.vals[2+4*4 + 20*(0+4*4)].re = +1; //substract 4h
    Dend_array.vals[3+4*4 + 20*(1+4*4)].re = +1; //substract 4h
    Dend_array.vals[0+4*3 + 20*(2+4*4)].re = -1; //add 4h
    Dend_array.vals[1+4*3 + 20*(3+4*4)].re = -1; //add 4h
    
    /* When sending in 5 from the right: go to 4h */
    /* Add 4h */
    Dend_array.vals[0+4*3 + 20*(0+4*5)].re = -1;
    Dend_array.vals[1+4*3 + 20*(1+4*5)].re = -1;
    Dend_array.vals[2+4*3 + 20*(2+4*5)].re = -1;
    Dend_array.vals[3+4*3 + 20*(3+4*5)].re = -1;
    /* Substract 4h */
    Dend_array.vals[0+4*4 + 20*(0+4*5)].re = +1;
    Dend_array.vals[1+4*4 + 20*(1+4*5)].re = +1;
    Dend_array.vals[2+4*4 + 20*(2+4*5)].re = +1;
    Dend_array.vals[3+4*4 + 20*(3+4*5)].re = +1;
    
    /* Right terminating array (for Dend) */
    /* 8th order */
    rightTerminator_array.vals[0].re = 0; //do nothing
    rightTerminator_array.vals[1].re = -4/(5*h);  //add at h
    rightTerminator_array.vals[2].re = +4/(5*h);  //substract at h
    rightTerminator_array.vals[3].re = -1/(5*h);  //add & substract at 2h
    rightTerminator_array.vals[4].re = +4/(105*h);//add & substract at 3h
    rightTerminator_array.vals[5].re = -1/(280*h);//add & substract at 4h
    
//    /* 2nd order */
//    rightTerminator_array.vals[0].re = 0;         //do nothing
//    rightTerminator_array.vals[1].re = -1/(2*h);  //add at h
//    rightTerminator_array.vals[2].re = +1/(2*h);  //substract at h
//    rightTerminator_array.vals[3].re = 0;         //add & substract at 2h
//    rightTerminator_array.vals[4].re = 0;         //add & substract at 3h
//    rightTerminator_array.vals[5].re = 0;         //add & substract at 4h
    
    /* Create the nodes */
    Dmid = tntNodeCreate(&Dmid_array, "DLUR", 4, 3, 4, 3);
    Dending = tntNodeCreate(&Dending_array, "DLUR", 4, 3, 4, 5);
    Dend = tntNodeCreate(&Dend_array, "DLUR", 4, 5, 4, 6);
    rightTerminator = tntNodeCreate(&rightTerminator_array, "LR", 6, 1);
    leftTerminator = tntNodeCreate(&leftTerminator_array, "LR", 1, 3);
    /* Use terminating nodes to produce starting and ending D nodes */
    /* First Dstart */
    Dmidc = tntNodeCopy(Dmid);
    tntNodeJoin(leftTerminator,"R",Dmidc,"L");
    Dstart = tntNodeContract(leftTerminator,Dmidc);
    /* Then Dend */
    tntNodeJoin(Dend,"R",rightTerminator,"L");
    Dend = tntNodeContract(rightTerminator,Dend);
    
    /* Produce the full network */
    /* First initialise it */
    d2 = tntNetworkCreate();
    
    /* Insert the bulk nodes */
    for(unsigned i = 0; i<L-3; i++){
        /* Create mid node at i */
        Dmidc = tntNodeCopy(Dmid);
        tntNodeInsertAtStart(Dmidc, "L", "R", d2);
    }
    
    /* Finally, insert the terminating nodes */
    tntNodeInsertAtStart(Dstart,"L","R",d2);
    tntNodeInsertAtEnd(Dending,"L","R",d2);
    tntNodeInsertAtEnd(Dend,"L","R",d2);

    /* Make tensors go from sparse to dense (and hence decrease the bond dimension) */
    dummy += tntMpoTruncate(d2, -1);
    
    /* Now free everything except d, Dmidc, the start, ending and end D nodes. */
    tntNodeFree(&Dmid);
    tntComplexArrayFree(&leftTerminator_array);
    tntComplexArrayFree(&Dmid_array);
    tntComplexArrayFree(&rightTerminator_array);

    /* Return d2 */
    return d2;
}

/*! \ingroup pde
 *
 * TOWRITE
 */
tntNetwork tntMpsPde2DCreateOrder2Diff1(char BC[], unsigned L){
    
    if(L < 5){
        tntErrorPrint("Cannot Create 1st order differencer when length of MPS is less than 5. Exiting."); /* NO_COVERAGE */
        exit(-1);
    }
    
    /* Declare necessary variables */
    tntNetwork dd1;
    tntNode Dstart, Dmid, Dending, Dend, leftTerminator, rightTerminator, Dmidc; //Needed to produce dd1.
    tntComplexArray leftTerminator_array, rightTerminator_array, Dending_array, Dend_array, Dmid_array; //Arrays used to produce D nodes
    double h = pow((double) 1/2, (double) L);
    double dummy;
    
    /* Initialise the start and end arrays to zeros */
    leftTerminator_array = tntComplexArrayAlloc(3);
    Dmid_array = tntComplexArrayAlloc(12,12); //bulk node
    Dending_array = tntComplexArrayAlloc(12,20); //second-to-last node
    Dend_array = tntComplexArrayAlloc(20,24); //last node
    rightTerminator_array = tntComplexArrayAlloc(6);
    
    /* Fill arrays with appropriate values. Indices are DL (fast multiindex) and UR. */
    /* Start with the left terminator array for Dstart */
    /* Depending on inputed boundary conditions, produce the left terminating node */
    if(!strcmp(BC, "Periodic") ){
        
        /* For periodic boundary conditions, left/right shifted wavefunctions at the boundaries just means they go to the other side of the domain. This can be done by filling the left terminator array with only 1s (as both an incoming 0 or 1 on the right leg is acceptable) */
        leftTerminator_array.vals[0].re = 1;
        leftTerminator_array.vals[1].re = 1;
        leftTerminator_array.vals[2].re = 1;
        
    }
    else if(!strcmp(BC,"Homogenous Dirichlet") ){
        
        /* For Homogenous Dirichlet boundary conditions, we need to make sure that the left/right shifted wavefunctions are zero whenever they're outside the domain. This can be done by letting the terminator start nodes take the value "0" when there's an incoming "1" on their right leg, as this means we're leaving the domain.  */
        leftTerminator_array.vals[0].re = 1;
    }
    else if(!strcmp(BC,"Periodic Reflective") ){
        
        /* In mirror BCs, the wavefunction is periodic, and the boundary constitutes an axis/plane of symmetry. The first derivative is always zero in this case, while the 2nd derivative equals 2*valueAtBoundary - 2*valueJustRightOfBoundary */
        leftTerminator_array.vals[0].re = 1;
        leftTerminator_array.vals[1].re = -1;
        leftTerminator_array.vals[2].re = -1;
    }
    else{
        tntErrorPrint("The boundary conditions (BC) inputed into tntMpsPde2DCreateProlongator are not supported. Exiting.");
        exit(0);
    }
    
    /* Create the bulk nodes and the two last nodes */
    /* Dmid */
    /* When sending in 0 from the right: do nothing */
    Dmid_array.vals[0 + 12*0].re = 1;
    Dmid_array.vals[1 + 12*1].re = 1;
    Dmid_array.vals[2 + 12*2].re = 1;
    Dmid_array.vals[3 + 12*3].re = 1;
    
    /* When sending in 1 from the right: add */
    Dmid_array.vals[1+4*0 + 12*(0+4*1)].re = 1;
    Dmid_array.vals[0+4*1 + 12*(1+4*1)].re = 1;
    Dmid_array.vals[3+4*0 + 12*(2+4*1)].re = 1; //01->11
    Dmid_array.vals[2+4*1 + 12*(3+4*1)].re = 1; //11->01+
    
    /* When sending in 2 from the right: substract */
    Dmid_array.vals[1+4*2 + 12*(0+4*2)].re = 1;
    Dmid_array.vals[0+4*0 + 12*(1+4*2)].re = 1;
    Dmid_array.vals[3+4*2 + 12*(2+4*2)].re = 1; //01->11-
    Dmid_array.vals[2+4*0 + 12*(3+4*2)].re = 1; //11->01
    
    /* Dending *
    /* When sending in 0 from the right: do nothing */
    Dending_array.vals[0 + 12*0].re = 1;
    Dending_array.vals[1 + 12*1].re = 1;
    Dending_array.vals[2 + 12*2].re = 1;
    Dending_array.vals[3 + 12*3].re = 1;
    
    /* When sending in 1 from the right: add */
    Dending_array.vals[1+4*0 + 12*(0+4*1)].re = 1;
    Dending_array.vals[0+4*1 + 12*(1+4*1)].re = 1;
    Dending_array.vals[3+4*0 + 12*(2+4*1)].re = 1;
    Dending_array.vals[2+4*1 + 12*(3+4*1)].re = 1;
    
    /* When sending in 2 from the right: substract */
    Dending_array.vals[1+4*2 + 12*(0+4*2)].re = 1;
    Dending_array.vals[0+4*0 + 12*(1+4*2)].re = 1;
    Dending_array.vals[3+4*2 + 12*(2+4*2)].re = 1;
    Dending_array.vals[2+4*0 + 12*(3+4*2)].re = 1;
    
    /* When sending in 3 from the right: add 4h */
    Dending_array.vals[0+4*1 + 12*(0+4*3)].re = 1; //add
    Dending_array.vals[1+4*1 + 12*(1+4*3)].re = 1; //add
    Dending_array.vals[2+4*1 + 12*(2+4*3)].re = 1; //add
    Dending_array.vals[3+4*1 + 12*(3+4*3)].re = 1; //add
    
    /* When sending in 4 from the right: substract 4h */
    Dending_array.vals[0+4*2 + 12*(0+4*4)].re = 1; //substract
    Dending_array.vals[1+4*2 + 12*(1+4*4)].re = 1; //substract
    Dending_array.vals[2+4*2 + 12*(2+4*4)].re = 1; //substract
    Dending_array.vals[3+4*2 + 12*(3+4*4)].re = 1; //substract
    
    /* Dend */
    /* When sending in 0 from the right: do nothing */
    Dend_array.vals[0 + 20*0].re = 1;
    Dend_array.vals[1 + 20*1].re = 1;
    Dend_array.vals[2 + 20*2].re = 1;
    Dend_array.vals[3 + 20*3].re = 1;
    
    /* When sending in 1 from the right: add */
    Dend_array.vals[1+4*0 + 20*(0+4*1)].re = 1;
    Dend_array.vals[0+4*1 + 20*(1+4*1)].re = 1;
    Dend_array.vals[3+4*0 + 20*(2+4*1)].re = 1;
    Dend_array.vals[2+4*1 + 20*(3+4*1)].re = 1;
    
    /* When sending in 2 from the right: substract */
    Dend_array.vals[1+4*2 + 20*(0+4*2)].re = 1;
    Dend_array.vals[0+4*0 + 20*(1+4*2)].re = 1;
    Dend_array.vals[3+4*2 + 20*(2+4*2)].re = 1;
    Dend_array.vals[2+4*0 + 20*(3+4*2)].re = 1;
    
    /* When sending in 3 from the right: go to 2h */
    Dend_array.vals[0+4*1 + 20*(0+4*3)].re = +1; //add
    Dend_array.vals[0+4*2 + 20*(0+4*3)].re = +1; //substract
    Dend_array.vals[1+4*1 + 20*(1+4*3)].re = +1; //add
    Dend_array.vals[1+4*2 + 20*(1+4*3)].re = +1; //substract
    Dend_array.vals[2+4*1 + 20*(2+4*3)].re = +1; //add
    Dend_array.vals[2+4*2 + 20*(2+4*3)].re = +1; //substract
    Dend_array.vals[3+4*1 + 20*(3+4*3)].re = +1; //add
    Dend_array.vals[3+4*2 + 20*(3+4*3)].re = +1; //substract
    
    /* When sending in 4 from the right: go to 3h */
    /* Perform the addition & substraction at this end node, and send instructions to next node */
    Dend_array.vals[1+4*1 + 20*(0+4*4)].re = +1; //add
    Dend_array.vals[0+4*2 + 20*(1+4*4)].re = +1; //substract
    Dend_array.vals[3+4*1 + 20*(2+4*4)].re = +1; //add
    Dend_array.vals[2+4*2 + 20*(3+4*4)].re = +1; //substract
    /* Special case when remainder must be transferred */
    Dend_array.vals[1+4*4 + 20*(0+4*4)].re = +1; //substract 4h
    Dend_array.vals[0+4*3 + 20*(1+4*4)].re = +1; //add 4h
    Dend_array.vals[3+4*4 + 20*(2+4*4)].re = +1; //substract 4h
    Dend_array.vals[2+4*3 + 20*(3+4*4)].re = +1; //add 4h
    
    /* When sending in 5 from the right: go to 4h */
    /* Add 4h */
    Dend_array.vals[0+4*3 + 20*(0+4*5)].re = +1;
    Dend_array.vals[1+4*3 + 20*(1+4*5)].re = +1;
    Dend_array.vals[2+4*3 + 20*(2+4*5)].re = +1;
    Dend_array.vals[3+4*3 + 20*(3+4*5)].re = +1;
    /* Substract 4h */
    Dend_array.vals[0+4*4 + 20*(0+4*5)].re = +1;
    Dend_array.vals[1+4*4 + 20*(1+4*5)].re = +1;
    Dend_array.vals[2+4*4 + 20*(2+4*5)].re = +1;
    Dend_array.vals[3+4*4 + 20*(3+4*5)].re = +1;
    
    /* Right terminating array (for Dend) */
    /* 8th order */
    rightTerminator_array.vals[0].re = -205/(72*h*h);//do nothing
    rightTerminator_array.vals[1].re = +8/(5*h*h);   //add at h
    rightTerminator_array.vals[2].re = +8/(5*h*h);   //substract at h
    rightTerminator_array.vals[3].re = -1/(5*h*h);   //add & substract at 2h
    rightTerminator_array.vals[4].re = +8/(315*h*h); //add & substract at 3h
    rightTerminator_array.vals[5].re = -1/(560*h*h); //add & substract at 4h
    
//    /* 2nd order */
//    rightTerminator_array.vals[0].re = -2/(h*h);         //do nothing
//    rightTerminator_array.vals[1].re = +1/(h*h);  //add at h
//    rightTerminator_array.vals[2].re = +1/(h*h);  //substract at h
//    rightTerminator_array.vals[3].re = 0;         //add & substract at 2h
//    rightTerminator_array.vals[4].re = 0;         //add & substract at 3h
//    rightTerminator_array.vals[5].re = 0;         //add & substract at 4h
    
    /* Create the nodes */
    Dmid = tntNodeCreate(&Dmid_array, "DLUR", 4, 3, 4, 3);
    Dending = tntNodeCreate(&Dending_array, "DLUR", 4, 3, 4, 5);
    Dend = tntNodeCreate(&Dend_array, "DLUR", 4, 5, 4, 6);
    rightTerminator = tntNodeCreate(&rightTerminator_array, "LR", 6, 1);
    leftTerminator = tntNodeCreate(&leftTerminator_array, "LR", 1, 3);
    /* Use terminating nodes to produce starting and ending D nodes */
    /* First Dstart */
    Dmidc = tntNodeCopy(Dmid);
    tntNodeJoin(leftTerminator,"R",Dmidc,"L");
    Dstart = tntNodeContract(leftTerminator,Dmidc);
    /* Then Dend */
    tntNodeJoin(Dend,"R",rightTerminator,"L");
    Dend = tntNodeContract(rightTerminator,Dend);
    
    /* Produce the full network */
    /* First initialise it */
    dd1 = tntNetworkCreate();
    
    /* Insert the bulk nodes */
    for(unsigned i = 0; i<L-3; i++){
        /* Create mid node at i */
        Dmidc = tntNodeCopy(Dmid);
        tntNodeInsertAtStart(Dmidc, "L", "R", dd1);
    }
    
    /* Finally, insert the terminating nodes */
    tntNodeInsertAtStart(Dstart,"L","R",dd1);
    tntNodeInsertAtEnd(Dending,"L","R",dd1);
    tntNodeInsertAtEnd(Dend,"L","R",dd1);

    /* Make tensors go from sparse to dense (and hence decrease the bond dimension) */
    dummy += tntMpoTruncate(dd1, -1);
    
    /* Now free everything except d, Dmidc, the start, ending and end D nodes. */
    tntNodeFree(&Dmid);
    tntComplexArrayFree(&leftTerminator_array);
    tntComplexArrayFree(&Dmid_array);
    tntComplexArrayFree(&rightTerminator_array);

    /* Return dd1 */
    return dd1;
}

/*! \ingroup pde
 *
 * TOWRITE
 */
tntNetwork tntMpsPde2DCreateOrder2Diff2(char BC[], unsigned L){
    
    if(L < 5){
        tntErrorPrint("Cannot Create 1st order differencer when length of MPS is less than 5. Exiting."); /* NO_COVERAGE */
        exit(-1);
    }
    
    /* Declare necessary variables */
    tntNetwork dd2;
    tntNode Dstart, Dmid, Dending, Dend, leftTerminator, rightTerminator, Dmidc; //Needed to produce dd2.
    tntComplexArray leftTerminator_array, rightTerminator_array, Dending_array, Dend_array, Dmid_array; //Arrays used to produce D nodes
    double h = pow((double) 1/2, (double) L);
    double dummy;
    
    /* Initialise the start and end arrays to zeros */
    leftTerminator_array = tntComplexArrayAlloc(3);
    Dmid_array = tntComplexArrayAlloc(12,12); //bulk node
    Dending_array = tntComplexArrayAlloc(12,20); //second-to-last node
    Dend_array = tntComplexArrayAlloc(20,24); //last node
    rightTerminator_array = tntComplexArrayAlloc(6);
    
    /* Fill arrays with appropriate values. Indices are DL (fast multiindex) and UR. */
    /* Start with the left terminator array for Dstart */
    /* Depending on inputed boundary conditions, produce the left terminating node */
    if(!strcmp(BC, "Periodic") ){
        
        /* For periodic boundary conditions, left/right shifted wavefunctions at the boundaries just means they go to the other side of the domain. This can be done by filling the left terminator array with only 1s (as both an incoming 0 or 1 on the right leg is acceptable) */
        leftTerminator_array.vals[0].re = 1;
        leftTerminator_array.vals[1].re = 1;
        leftTerminator_array.vals[2].re = 1;
        
    }
    else if(!strcmp(BC,"Homogenous Dirichlet") ){
        
        /* For Homogenous Dirichlet boundary conditions, we need to make sure that the left/right shifted wavefunctions are zero whenever they're outside the domain. This can be done by letting the terminator start nodes take the value "0" when there's an incoming "1" on their right leg, as this means we're leaving the domain.  */
        leftTerminator_array.vals[0].re = 1;
    }
    else if(!strcmp(BC,"Periodic Reflective") ){
        
        /* In mirror BCs, the wavefunction is periodic, and at the boundary constitutes an axis/plane of symmetry. The first derivative is always zero in this case, while the 2nd derivative equals 2*valueAtBoundary - 2*valueJustRightOfBoundary */
        leftTerminator_array.vals[0].re = 1;
        leftTerminator_array.vals[1].re = -1;
        leftTerminator_array.vals[2].re = -1;
    }
    else{
        tntErrorPrint("The boundary conditions (BC) inputed into tntMpsPde2DCreateProlongator are not supported. Exiting.");
        exit(0);
    }
    
    /* Create the bulk nodes and the two last nodes */
    /* Dmid */
    /* When sending in 0 from the right: do nothing */
    Dmid_array.vals[0 + 12*0].re = 1;
    Dmid_array.vals[1 + 12*1].re = 1;
    Dmid_array.vals[2 + 12*2].re = 1;
    Dmid_array.vals[3 + 12*3].re = 1;
    
    /* When sending in 1 from the right: add */
    Dmid_array.vals[2+4*0 + 12*(0+4*1)].re = 1;
    Dmid_array.vals[3+4*0 + 12*(1+4*1)].re = 1;
    Dmid_array.vals[0+4*1 + 12*(2+4*1)].re = 1;
    Dmid_array.vals[1+4*1 + 12*(3+4*1)].re = 1;
    
    /* When sending in 2 from the right: substract */
    Dmid_array.vals[2+4*2 + 12*(0+4*2)].re = 1;
    Dmid_array.vals[3+4*2 + 12*(1+4*2)].re = 1;
    Dmid_array.vals[0+4*0 + 12*(2+4*2)].re = 1;
    Dmid_array.vals[1+4*0 + 12*(3+4*2)].re = 1;
    
    /* Dending *
    /* When sending in 0 from the right: do nothing */
    Dending_array.vals[0 + 12*0].re = 1;
    Dending_array.vals[1 + 12*1].re = 1;
    Dending_array.vals[2 + 12*2].re = 1;
    Dending_array.vals[3 + 12*3].re = 1;
    
    /* When sending in 1 from the right: add */
    Dending_array.vals[2+4*0 + 12*(0+4*1)].re = 1;
    Dending_array.vals[3+4*0 + 12*(1+4*1)].re = 1;
    Dending_array.vals[0+4*1 + 12*(2+4*1)].re = 1;
    Dending_array.vals[1+4*1 + 12*(3+4*1)].re = 1;
    
    /* When sending in 2 from the right: substract */
    Dending_array.vals[2+4*2 + 12*(0+4*2)].re = 1;
    Dending_array.vals[3+4*2 + 12*(1+4*2)].re = 1;
    Dending_array.vals[0+4*0 + 12*(2+4*2)].re = 1;
    Dending_array.vals[1+4*0 + 12*(3+4*2)].re = 1;
    
    /* When sending in 3 from the right: add 4h */
    Dending_array.vals[0+4*1 + 12*(0+4*3)].re = 1; //add
    Dending_array.vals[1+4*1 + 12*(1+4*3)].re = 1; //add
    Dending_array.vals[2+4*1 + 12*(2+4*3)].re = 1; //add
    Dending_array.vals[3+4*1 + 12*(3+4*3)].re = 1; //add
    
    /* When sending in 4 from the right: substract 4h */
    Dending_array.vals[0+4*2 + 12*(0+4*4)].re = 1; //substract
    Dending_array.vals[1+4*2 + 12*(1+4*4)].re = 1; //substract
    Dending_array.vals[2+4*2 + 12*(2+4*4)].re = 1; //substract
    Dending_array.vals[3+4*2 + 12*(3+4*4)].re = 1; //substract
    
    /* Dend */
    /* When sending in 0 from the right: do nothing */
    Dend_array.vals[0 + 20*0].re = 1;
    Dend_array.vals[1 + 20*1].re = 1;
    Dend_array.vals[2 + 20*2].re = 1;
    Dend_array.vals[3 + 20*3].re = 1;
    
    /* When sending in 1 from the right: add */
    Dend_array.vals[2+4*0 + 20*(0+4*1)].re = 1;
    Dend_array.vals[3+4*0 + 20*(1+4*1)].re = 1;
    Dend_array.vals[0+4*1 + 20*(2+4*1)].re = 1;
    Dend_array.vals[1+4*1 + 20*(3+4*1)].re = 1;
    
    /* When sending in 2 from the right: substract */
    Dend_array.vals[2+4*2 + 20*(0+4*2)].re = 1;
    Dend_array.vals[3+4*2 + 20*(1+4*2)].re = 1;
    Dend_array.vals[0+4*0 + 20*(2+4*2)].re = 1;
    Dend_array.vals[1+4*0 + 20*(3+4*2)].re = 1;
    
    /* When sending in 3 from the right: go to 2h */
    Dend_array.vals[0+4*1 + 20*(0+4*3)].re = +1; //add
    Dend_array.vals[0+4*2 + 20*(0+4*3)].re = +1; //substract
    Dend_array.vals[1+4*1 + 20*(1+4*3)].re = +1; //add
    Dend_array.vals[1+4*2 + 20*(1+4*3)].re = +1; //substract
    Dend_array.vals[2+4*1 + 20*(2+4*3)].re = +1; //add
    Dend_array.vals[2+4*2 + 20*(2+4*3)].re = +1; //substract
    Dend_array.vals[3+4*1 + 20*(3+4*3)].re = +1; //add
    Dend_array.vals[3+4*2 + 20*(3+4*3)].re = +1; //substract
    
    /* When sending in 4 from the right: go to 3h */
    /* Perform the addition & substraction at this end node, and send instructions to next node */
    Dend_array.vals[2+4*1 + 20*(0+4*4)].re = +1; //add
    Dend_array.vals[3+4*1 + 20*(1+4*4)].re = +1; //add
    Dend_array.vals[0+4*2 + 20*(2+4*4)].re = +1; //substract
    Dend_array.vals[1+4*2 + 20*(3+4*4)].re = +1; //substract
    /* Special case when remainder must be transferred */
    Dend_array.vals[2+4*4 + 20*(0+4*4)].re = +1; //substract 4h
    Dend_array.vals[3+4*4 + 20*(1+4*4)].re = +1; //substract 4h
    Dend_array.vals[0+4*3 + 20*(2+4*4)].re = +1; //add 4h
    Dend_array.vals[1+4*3 + 20*(3+4*4)].re = +1; //add 4h
    
    /* When sending in 5 from the right: go to 4h */
    /* Add 4h */
    Dend_array.vals[0+4*3 + 20*(0+4*5)].re = +1;
    Dend_array.vals[1+4*3 + 20*(1+4*5)].re = +1;
    Dend_array.vals[2+4*3 + 20*(2+4*5)].re = +1;
    Dend_array.vals[3+4*3 + 20*(3+4*5)].re = +1;
    /* Substract 4h */
    Dend_array.vals[0+4*4 + 20*(0+4*5)].re = +1;
    Dend_array.vals[1+4*4 + 20*(1+4*5)].re = +1;
    Dend_array.vals[2+4*4 + 20*(2+4*5)].re = +1;
    Dend_array.vals[3+4*4 + 20*(3+4*5)].re = +1;
    
    /* Right terminating array (for Dend) */
    /* 8th order */
    rightTerminator_array.vals[0].re = -205/(72*h*h);//do nothing
    rightTerminator_array.vals[1].re = +8/(5*h*h);   //add at h
    rightTerminator_array.vals[2].re = +8/(5*h*h);   //substract at h
    rightTerminator_array.vals[3].re = -1/(5*h*h);   //add & substract at 2h
    rightTerminator_array.vals[4].re = +8/(315*h*h); //add & substract at 3h
    rightTerminator_array.vals[5].re = -1/(560*h*h); //add & substract at 4h
    
//    /* 2nd order */
//    rightTerminator_array.vals[0].re = -2/(h*h);         //do nothing
//    rightTerminator_array.vals[1].re = +1/(h*h);  //add at h
//    rightTerminator_array.vals[2].re = +1/(h*h);  //substract at h
//    rightTerminator_array.vals[3].re = 0;         //add & substract at 2h
//    rightTerminator_array.vals[4].re = 0;         //add & substract at 3h
//    rightTerminator_array.vals[5].re = 0;         //add & substract at 4h
    
    /* Create the nodes */
    Dmid = tntNodeCreate(&Dmid_array, "DLUR", 4, 3, 4, 3);
    Dending = tntNodeCreate(&Dending_array, "DLUR", 4, 3, 4, 5);
    Dend = tntNodeCreate(&Dend_array, "DLUR", 4, 5, 4, 6);
    rightTerminator = tntNodeCreate(&rightTerminator_array, "LR", 6, 1);
    leftTerminator = tntNodeCreate(&leftTerminator_array, "LR", 1, 3);
    /* Use terminating nodes to produce starting and ending D nodes */
    /* First Dstart */
    Dmidc = tntNodeCopy(Dmid);
    tntNodeJoin(leftTerminator,"R",Dmidc,"L");
    Dstart = tntNodeContract(leftTerminator,Dmidc);
    /* Then Dend */
    tntNodeJoin(Dend,"R",rightTerminator,"L");
    Dend = tntNodeContract(rightTerminator,Dend);
    
    /* Produce the full network */
    /* First initialise it */
    dd2 = tntNetworkCreate();
    
    /* Insert the bulk nodes */
    for(unsigned i = 0; i<L-3; i++){
        /* Create mid node at i */
        Dmidc = tntNodeCopy(Dmid);
        tntNodeInsertAtStart(Dmidc, "L", "R", dd2);
    }
    
    /* Finally, insert the terminating nodes */
    tntNodeInsertAtStart(Dstart,"L","R",dd2);
    tntNodeInsertAtEnd(Dending,"L","R",dd2);
    tntNodeInsertAtEnd(Dend,"L","R",dd2);

    /* Make tensors go from sparse to dense (and hence decrease the bond dimension) */
    dummy += tntMpoTruncate(dd2, -1);
    
    /* Now free everything except dd2, Dmidc, the start, ending and end D nodes. */
    tntNodeFree(&Dmid);
    tntComplexArrayFree(&leftTerminator_array);
    tntComplexArrayFree(&Dmid_array);
    tntComplexArrayFree(&rightTerminator_array);

    /* Return dd2 */
    return dd2;
}
