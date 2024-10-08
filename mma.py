import torch


# Volta: 8x4 4x4
# Ampere: 16x8 16x16

# Ampere dimension simplification: Objetive 4x2 or 4x4
# Volta available ops: 2x1 or 2x2

# Define 4x2 target matrix
goal42 = torch.tensor([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8]], dtype=torch.float32)

# A 4x2 matrix is equivalent to a 2x2x2 tensor
# Define 2x2 operation matrix
op1_22 = torch.tensor([[1,2],
                  [3,4]], dtype=torch.float32)

# Define 2x2 operation matrix
op2_22 = torch.tensor([[5,6],
                  [7,8]], dtype=torch.float32)

# Concat result 
result42 = torch.concat((op1_22, op2_22),0)
#print(result42)

# Define 4x4 target matrix
goal42 = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]], dtype=torch.float32)

# A 4x4 matrix is equivalent to 4x2x2 tensor
# Define 2x2 operation matrix
op1_00 = torch.tensor([[1,2],
                  [5,6]], dtype=torch.float32)

# Define 2x2 operation matrix
op2_10 = torch.tensor([[9,10],
                  [13,14]], dtype=torch.float32)

# Define 2x2 operation matrix
op3_01 = torch.tensor([[3,4],
                  [7,8]], dtype=torch.float32)

# Define 2x2 operation matrix
op4_11 = torch.tensor([[11,12],
                  [15,16]], dtype=torch.float32)


# Concat result 
result42_1 = torch.concat((op1_00, op2_10),0)
result42_2 = torch.concat((op3_01, op4_11),0)
result44 = torch.concat((result42_1, result42_2),1)

print(result42_1)
print(result42_2)
print(result44)

"""
There's a relation over these matrices. As of now, It's not useful to use 4x4 as its store less data (16)
16x8 (128) and 16x16 (256) can both be reduced using 4x8x4 (128) and 8x8x4 (256) tensors.
"""
