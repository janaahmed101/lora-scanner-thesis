"""""""""""""""""""""""""""""""""""""""""""""""
Created on Wed Jun 20 07:56:00 2018
Decoder for Nominal LoRa and Interleaved-Chirp 
Spreading LoRa Modulation
@author: tallal
"""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
from sympy.combinatorics.graycode import bin_to_gray
from LoRa_Rx_SDR import lora_sdr_config as lora_cfg

Last_Decoded_PHY = b""
Last_Decoded_MAC = b""
DEBUG_TRACE = lora_cfg.LORAWAN_DEBUG

# =============================================================================
#  Generator of (8,4) Hamming Code , H_CR4 (i.e., Parity Check Matrix)
# =============================================================================
G     = np.array([[1, 0, 0, 0, 1, 0, 1, 1], [ 0, 1, 0, 0, 1, 1, 1, 0], [0, 0, 1, 0, 1, 1, 0, 1],[ 0, 0, 0, 1, 0, 1, 1, 1]]);
H_CR4 = np.transpose(G[:,4:8])
H_CR4 = np.hstack((H_CR4,np.eye(4)))
H_CR3 = H_CR4[0:3,0:7]

G           = np.transpose(G) 
H_CR4_Str_1 = ""
Syndrome_Table_CR4 = np.zeros((8))
      
H_CR3_Str_1=""
Syndrome_Table_CR3=np.zeros((np.shape(H_CR3)[1]))
                     
# =============================================================================
# Syndrome & Correction Tables for CR4
# =============================================================================
H_CR4 = np.flipud(H_CR4) #this flip is done because we are reading from right to left in matlab 
H_CR4 = np.transpose(H_CR4);

for i in range (0,8):
    H_CR4_Str_1 = "";
    for j in range (0,(np.shape(H_CR4)[1])):
        H_CR4_Str_1 = H_CR4_Str_1 + str(int(H_CR4[i,j]));   
    Syndrome_Table_CR4[i]=int((H_CR4_Str_1),2);
   
Correction_Table_CR4=np.zeros((16,8));
for i in range (0,8):
    Correction_Table_CR4[int(Syndrome_Table_CR4[i]),i]=1;

# =============================================================================
# Syndrome & Correction Tables for CR3
#=============================================================================
H_CR3 = np.flipud(H_CR3) #this flip is done because we are reading from right to left in matlab 
H_CR3 = np.transpose(H_CR3);

for i in range (0,7):
    H_CR3_Str_1 = "";
    for j in range (0,(np.shape(H_CR3)[1])):
        H_CR3_Str_1 = H_CR3_Str_1 + str(int(H_CR3[i,j]));   
    Syndrome_Table_CR3[i]=int((H_CR3_Str_1),2);
   
Correction_Table_CR3=np.zeros((8,7));
for i in range (0,7):
    Correction_Table_CR3[int(Syndrome_Table_CR3[i]),i]=1;

# =============================================================================
# Generator of the Header CRC
# The Header CRC works as follows
# 1. Calculate Parity bits from 8 bits of 2nd and 3rd columns
# 2. XOR 1st column with the parity bits to formulate the 5th column
# 3. 1st bit in 4th column is mod(sum(1st column)
# Note: The remaining bits of 4th column are zeros
# =============================================================================
G_HCRC =np.array( [[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
                        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]]); 
         
# =============================================================================
# Whitening Sequence of LoRa and Gray Map
# Generator X9 + X8 + X7 +X4 + 1
# =============================================================================
Nof_Outputs    = 262; # Selected to Limit to Max LoRa Packet size of 255 Bytes
Reg_Content    = np.ones(9);
Whitening_Seq  = np.zeros(Nof_Outputs);
Reg_Content[0] = 0
for Output in range( 0, Nof_Outputs):
    Whitening_Seq[Output] = Reg_Content[np.size(Reg_Content)-1];
    Reg_Content           = np.append(np.mod((Reg_Content[8]+Reg_Content[7]+Reg_Content[6]+Reg_Content[3]+Reg_Content[0]),2),Reg_Content[0:np.size(Reg_Content)-1]);
Whitening_Bytes = np.zeros(255)
for i in range (0,255):
    Binary = "";
    for j in range (i,i+8):
        Binary = Binary+str(int(Whitening_Seq[j]));
    Whitening_Bytes[i] = int((Binary),2);
Whitening_Bytes = Whitening_Bytes.astype(int);


# =============================================================================
# Extract decoded packet bytes
# =============================================================================
def LoRa_Bits_To_Bytes(LoRa_Dec_Bits, Nof_Bytes):
    Dec_Bits_Final_Reshape = np.zeros((8,Nof_Bytes));

    counter = 0
    for i in range (0,Nof_Bytes*2,2):
        Dec_Bits_Final_Reshape[0:4,counter] = LoRa_Dec_Bits[:,i];
        j=i+1;
        Dec_Bits_Final_Reshape[4:8,counter] = LoRa_Dec_Bits[:,j];
        counter=counter+1;

    Dec_Bits_Final_Reshape = Dec_Bits_Final_Reshape.astype(int);
    Dec_Bits_Final_Reshape = np.fliplr(np.transpose(Dec_Bits_Final_Reshape));

    Dec_Bytes = np.zeros(Nof_Bytes, dtype=np.uint8);
    for i in range (0,Nof_Bytes):
      X="";
      for j in range(0,8):
        X=X+str(Dec_Bits_Final_Reshape[i,j]);
      Dec_Bytes[i]=int(X,2);

    return bytes(Dec_Bytes.tolist())

def Get_Decoded_Bytes():
    return Last_Decoded_PHY, Last_Decoded_MAC

# =============================================================================
# Decoding ASCII 
# =============================================================================
def ASCIII_Decoding(LoRa_Dec_Bits,Nof_Bytes):
    
    Dec_Bits_Final_Reshape = np.zeros((8,Nof_Bytes));

    counter = 0
    for i in range (0,Nof_Bytes*2,2):
        Dec_Bits_Final_Reshape[0:4,counter] = LoRa_Dec_Bits[:,i];
        j=i+1;
        Dec_Bits_Final_Reshape[4:8,counter] = LoRa_Dec_Bits[:,j];
        counter=counter+1;

    X="";
    Dec_Bits_Final_Reshape = Dec_Bits_Final_Reshape.astype(int);
    Dec_Bytes              = np.zeros(Nof_Bytes);   
    Ascii_Array_Rxed       = np.zeros(Nof_Bytes).astype(str);
    Removed_Rxed_Array     = np.zeros(4).astype(str);
    Ascii_Array_Rxed_New   = np.zeros(Nof_Bytes-4).astype(str);
    Dec_Bits_Final_Reshape = np.fliplr(np.transpose(Dec_Bits_Final_Reshape));

    for i in range (0,Nof_Bytes): 
      X="";
      for j in range(0,8):
        X=X+str(Dec_Bits_Final_Reshape[i,j]);
      Dec_Bytes[i]=int(X,2);
      Ascii_Array_Rxed[i]=chr( int(Dec_Bytes[i])) 
    #print(Dec_Bytes)
    for j in range (0,4):
        Removed_Rxed_Array[j]= Ascii_Array_Rxed[j];
    
    counter=0;
    for k in range (4,Nof_Bytes):
        Ascii_Array_Rxed_New[counter]=Ascii_Array_Rxed[k];
        counter=counter+1;
    if DEBUG_TRACE:
        print(Ascii_Array_Rxed_New)

# =============================================================================
# Decoding Header 
# =============================================================================
def Get_Header_Info(SF,Header_Syms):
    
    # =============================================================================
    # Generating Gray Map
    # =============================================================================
    Gray_Map =np.zeros((2**SF));
    for i in range(0,2**SF):
        Gray_Map[i]=int((bin_to_gray(str(np.binary_repr(i)))),2);
                    
    # =============================================================================
    #Inverse Gray Mapping Header Bits    
    # =============================================================================
    Header_Bits = np.zeros((8,SF))
    for i in range(0,8):
        #Header_Dec[i]=int((bin_to_gray(str(np.binary_repr(int(Header_Syms[i]))))),2);
        Header_Dec = int(Header_Syms[i]-1);
        Header_Bits[i,:] = np.flipud(list(np.binary_repr((int(Gray_Map[Header_Dec])),SF)));
    Header_Bits = Header_Bits[:,2:np.shape(Header_Bits)[1]];
     
    # =============================================================================
    # Interleaving of Header Bits    
    # =============================================================================
    for Row in range(0,8):
        Header_Bits[Row,:]=np.roll(Header_Bits[Row,:],Row-0);
    #print(Header_Bits) 
     
    ## =============================================================================
    ## Header Syndrome and Error Correction if Needed
    ## =============================================================================
    Header_Syndrome=(np.mod((np.transpose(Header_Bits)@np.fliplr(H_CR4)),2)).astype(int);# if all are zeros then this is error free
    
    if np.sum(Header_Syndrome)>0:
        Correction_Bits=np.zeros((np.shape(Header_Syndrome)[0],8))
        for i in range(0,np.shape(Header_Syndrome)[0]):
            X="";
            for j in range(0,np.shape(Header_Syndrome)[1]):
                X=X+str(Header_Syndrome[i,j]);   
            Correction_Bits[i,:] = Correction_Table_CR4[int((X),2),:];
        Header_Bits=np.logical_xor(Header_Bits,np.transpose(Correction_Bits))*1;
    
    Header_Bits = Header_Bits[0:4,:];
    #print(Header_Bits)
    
    ## =============================================================================
    ## Checking Header CRC that works as follows
    ## 1. Calculate Parity Bits from 8 bits of 2nd and 3rd columns
    ## 2. XOR 1st column with the parity bits to formulate the 5th column
    ## 3. 1st bit of 4th column is modulo sum of 1st column
    ## Note: The remaining three bits of the 4th colummn are zero
    ## =============================================================================
    HCRC_Bits        = np.zeros((4,2))
    HCRC_Bits[0,0]   = np.mod(int(np.sum(Header_Bits[:,0])),2)
    HCRC_Computation = Header_Bits[:,1:3]
    HCRC_Computation = np.transpose(HCRC_Computation)
    HCRC_Computation = HCRC_Computation.reshape(1,8)
    HCRC_Computation = np.mod(np.matmul(HCRC_Computation,G_HCRC),2)
    HCRC_Computation = np.mod(Header_Bits[:,0]+HCRC_Computation[0,8:12],2)
    HCRC_Bits[:,1]   = HCRC_Computation

    # [MODDBG] Added debug marker: parse header fields before CRC check for debug visibility.
    Length_Bits_pre = Header_Bits[:,0:2]
    Length_Bits_pre = np.flipud(Length_Bits_pre)
    Length_Bits_pre = np.reshape(Length_Bits_pre,(1,8),'F')
    X_pre = ""
    for i in range(0,8):
        X_pre = X_pre + str(int(Length_Bits_pre[0,i]))
    PL_pre = int(X_pre,2)

    CR_Bits_pre = np.reshape(Header_Bits[1:4,2],(1,3))
    CR_Bits_pre = CR_Bits_pre.astype(int)
    CR_Bits_pre = np.fliplr(CR_Bits_pre)
    CR_Bits_Str_pre = ""
    for i in range(0,3):
        CR_Bits_Str_pre = CR_Bits_Str_pre + str(CR_Bits_pre[0,i])
    CR_pre = int(CR_Bits_Str_pre,2)
    CRC_pre = int(Header_Bits[0,2])
    if DEBUG_TRACE:
        print(f"[Decode_Symbols] Header parsed pre-CRC: PL={PL_pre}, CR={CR_pre}, CRC={CRC_pre}, SF={SF}")
    
    CRC_Check = np.mod(HCRC_Bits+Header_Bits[:,3:5],2)
    #print(HCRC_Bits, "CRC Check",CRC_Check)
    if int(np.sum(CRC_Check)==0):
        
        # =============================================================================
        # Parsing Header Bits
        # =============================================================================
        Length_Bits = Header_Bits[:,0:2];
        Length_Bits = np.flipud(Length_Bits);
        Length_Bits = np.reshape(Length_Bits,(1,8),'F')
        #Length_Bits = np.fliplr(Length_Bits);
        
        X="";
        for i in range(0,8):
            X=X+str(int(Length_Bits[0,i]));
        
        PL = int(X,2)
        
        CR_Bits     = np.reshape(Header_Bits[1:4,2],(1,3))        
        CR_Bits_Str = "";
        CR_Bits     = CR_Bits.astype(int)
        CR_Bits     = np.fliplr(CR_Bits)
        
        for i in range(0,3):
            CR_Bits_Str=CR_Bits_Str+str(CR_Bits[0,i])
        CR = int(CR_Bits_Str,2)
        CR = int(CR)
        CRC = int(Header_Bits[0,2])
        
        #Nof_LoRa_Syms = 8+np.ceil((PL*8+28-4*SF+16*CRC)/(4*SF))*(4+CR);
        #print(Header_Bits[:,5:SF-2])
        if SF>7:
            return(PL,CR,CRC,Header_Bits[:,5:SF-2],True)
        else:
            return(PL,CR,CRC,[],True)
    else:
        return([],[],[],[],False) 

# =============================================================================
# Decoding LoRa Packet
# =============================================================================
def Get_Data_Info(SF,CR,PL,CRC,Nof_LoRa_Syms,Rx_Sym,Header_Bits):
    global Last_Decoded_PHY, Last_Decoded_MAC
    
    # print('Passed Header_Bits', Header_Bits)
    # =============================================================================
    # Generating Gray Map
    # =============================================================================
    Gray_Map =np.zeros((2**SF));
    for i in range(0,2**SF):
        Gray_Map[i]=int((bin_to_gray(str(np.binary_repr(i)))),2);
    
    # =============================================================================
    # Decoding Rx Blocks
    # =============================================================================
    Nof_Blocks     = int(np.ceil((Nof_LoRa_Syms)/(4+CR)))
    Nof_Syms       = 4+CR
    #print(Nof_Blocks,Nof_LoRa_Syms)
    Rx_Sym_Reshape = np.zeros((int(np.size(Rx_Sym)/Nof_Blocks),int(Nof_Blocks)));
    
    Block_Bits = np.zeros((int(np.size(Rx_Sym)/Nof_Blocks),SF));
    for Block in range(0,Nof_Blocks):
        Rx_Sym_Reshape[:,Block] = Rx_Sym[Block*Nof_Syms:(Block+1)*Nof_Syms]

    Dec_Bits = np.zeros((4,Nof_Blocks*SF));
    
    for Block in range(0,Nof_Blocks):    
         Block_Sym = Rx_Sym_Reshape[:,Block];
    
         for i in range(0,int(np.size(Rx_Sym)/Nof_Blocks)):
             if Block_Sym[i]==0:
                 # A demodulated payload symbol of value 0 corresponds to gray index
                 # (s-1) = -1 == 2**SF - 1.  The original code hard-coded 128 (= 2**7),
                 # which silently corrupts every zero-valued symbol for any SF != 7.
                 Block_Sym[i]=2**SF;
             Block_Bits[i,:] = np.flipud(list(np.binary_repr(int(Gray_Map[int(Block_Sym[i])-1]),SF)));
         for Row in range (0,CR+4):
             Block_Bits[Row,:]=np.roll(Block_Bits[Row,:],Row-0);
         
         Dec_Bits[:,(Block)*SF:((Block+1)*SF)] = Block_Bits[0:4,:];
         
         # =============================================================================
         # if CR=3  
         # =============================================================================
         if CR == 3:
             Block_Syndrome  = (np.mod((np.transpose(Block_Bits)@H_CR3),2)).astype(int);# if all are zeros then this is error free
             Correction_Bits = np.zeros((np.shape(Block_Syndrome)[0],np.shape(Correction_Table_CR3)[1]));
             
             for i in range(0,SF):
                 #X="";
                 #for j in range(0,np.shape(Block_Syndrome)[1]):
                 #    X=X+str(Block_Syndrome[i,j]);   
                 #Correction_Bits[i,:] = Correction_Table_CR3[int((X),2),:];
                 #Block_Bits = np.logical_xor(Block_Bits,np.transpose(Correction_Bits))*1;
                 Correction_Index = Block_Syndrome[i,:]*2**np.arange(CR)[::-1]
                 Correction_Index = np.sum(Correction_Index)
                 Correction_Bits[i,:] = Correction_Table_CR3[Correction_Index,:]
             Block_Bits = np.logical_xor(Block_Bits,np.transpose(Correction_Bits))*1;
             Dec_Bits[:,(Block)*SF:((Block+1)*SF)] = Block_Bits[0:4,:];
         
         # =============================================================================
         # if CR=4
         # =============================================================================
         if CR==4:
             Block_Syndrome  = (np.mod((np.transpose(Block_Bits)@np.fliplr(H_CR4)),2)).astype(int);# if all are zeros then this is error free
             Correction_Bits = np.zeros((np.shape(Block_Syndrome)[0],np.shape(Correction_Table_CR4)[1]));
             
             for i in range(0,SF):
                 #X="";
                 #for j in range(0,np.shape(H_CR4)[1]):
                 #    X=X+str(Block_Syndrome[i,j]);
                 #Correction_Bits[i,:] = Correction_Table_CR4[int((X),2),:]
                 Correction_Index = Block_Syndrome[i,:]*2**np.arange(CR)[::-1]
                 Correction_Index = np.sum(Correction_Index)
                 Correction_Bits[i,:] = Correction_Table_CR4[Correction_Index,:]
             Block_Bits = np.logical_xor(Block_Bits,np.transpose(Correction_Bits))*1;
             Dec_Bits[:,(Block)*SF:((Block+1)*SF)] = Block_Bits[0:4,:];
    #print("Dec Bits Shape", np.shape(Dec_Bits))
    
    # =============================================================================
    # Adding Bits Transmitted within Header 
    # =============================================================================
    if SF>7:
        Bits_in_Header = Header_Bits;
        Dec_Bits_New   = Bits_in_Header;
        for i in range(0,np.shape(Dec_Bits)[1]):
            Dec_Bits_New = np.column_stack((Dec_Bits_New,Dec_Bits[:,i]));
    
    # =============================================================================
    # Dewhiteniing
    # =============================================================================
    Nof_Whitening_Bits         = PL*8;
    Nof_Whitening_Bytes        = int(np.ceil(Nof_Whitening_Bits/8));
    Pkt_Whitening_Bytes        = Whitening_Bytes[0:Nof_Whitening_Bytes];
    Pkt_Whitening_Bits         = np.zeros((8,PL));#8 is constant as 255 is changed into 8 bits
    Pkt_Whitening_Bits_columns = 2*PL;
    Pkt_Whitening_Bits_LoRa    = np.zeros((4,Pkt_Whitening_Bits_columns));    
    #print(Pkt_Whitening_Bytes)

    for Byte_Index in range (0,PL):
        Pkt_Whitening_Bits[:,Byte_Index]=np.flipud(list(np.binary_repr(int(Pkt_Whitening_Bytes[Byte_Index]),8)));#correct
        First  = np.transpose(Pkt_Whitening_Bits[0:4,Byte_Index]);
        Second = np.transpose(Pkt_Whitening_Bits[4:8,Byte_Index]);
        Pkt_Whitening_Bits_LoRa[:,2*Byte_Index:2*(Byte_Index+1)] = np.column_stack((First,Second));
    
    
    if SF>7:
        Dec_Bits_Final = np.logical_xor(Dec_Bits_New[:,0:PL*2],Pkt_Whitening_Bits_LoRa)*1;
        CRC_Bits = Dec_Bits_New[:,PL*2:PL*2+4]
    else:
        
        #print(np.shape(Dec_Bits[:,0:PL*2]),np.shape(Pkt_Whitening_Bits_LoRa))
        Dec_Bits_Final = np.mod(Dec_Bits[:,0:PL*2]+Pkt_Whitening_Bits_LoRa,2);
        #Dec_Bits_Final = np.logical_xor(Dec_Bits[:,0:PL*2],Pkt_Whitening_Bits_LoRa)*1;
        
        CRC_Bits = Dec_Bits[:,PL*2:PL*2+4]
    CRC_Bits = CRC_Bits.astype(int)
    
    
    if CRC ==1:
        #==============
        # Computing CRC
        #==============
        CRC_Polynomial    = [1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1]
        CRC_Msg_Bin       = np.transpose(np.reshape(np.transpose(Dec_Bits_Final),(PL,8)))
        CRC_Msg_Bin       = np.flipud(CRC_Msg_Bin)
        CRC_Msg_Bin       = np.reshape(np.transpose(CRC_Msg_Bin),(1,PL*8))
        CRC_Msg_Bin       = np.fliplr(CRC_Msg_Bin)
        Polynomial_Degree = np.max(np.where(CRC_Msg_Bin==1))
        Current_Degree    = Polynomial_Degree 
        Msg_Polynomial    = CRC_Msg_Bin
        
        while (Current_Degree>15):
            Msg_Polynomial[0,Current_Degree-16:Current_Degree+1] = np.logical_xor(CRC_Polynomial,Msg_Polynomial[0,Current_Degree-16:Current_Degree+1])
            Current_Degree = np.max(np.where(Msg_Polynomial==1))
        
        CRC_Bits_Computed = Msg_Polynomial[0,0:16]
        CRC_Bits_Computed = np.reshape(CRC_Bits_Computed,(4,4))
        CRC_Bits_Computed = np.transpose(CRC_Bits_Computed)
        
        if (np.sum(np.mod(CRC_Bits+CRC_Bits_Computed,2))==0):
            Last_Decoded_PHY = LoRa_Bits_To_Bytes(Dec_Bits_Final, PL)
            Last_Decoded_MAC = Last_Decoded_PHY[1:-4] if len(Last_Decoded_PHY) >= 5 else b""
            if DEBUG_TRACE:
                print('Successful_CRC')
            return(True)
        else:
            Last_Decoded_PHY = b""
            Last_Decoded_MAC = b""
            if DEBUG_TRACE:
                print('Failed_CRC')
            return(False)
    else:
        Last_Decoded_PHY = LoRa_Bits_To_Bytes(Dec_Bits_Final, PL)
        Last_Decoded_MAC = Last_Decoded_PHY[1:-4] if len(Last_Decoded_PHY) >= 5 else b""
        return(True)
