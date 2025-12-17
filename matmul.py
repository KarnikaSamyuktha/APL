def matrix_multiply(matrix1, matrix2):
    #to check if any empty matrix
    if not matrix1 or not matrix2:
        raise ValueError("Empty Matrix")
        
    #dimensions of matrix
    row1=len(matrix1)#row length of matrix1
    row2=len(matrix2)#row length of matrix2
    col1=len(matrix1[0])#column length of matrix1
    col2=len(matrix2[0])#column length of matrix2

    result_matrix = [[0 for _ in range(col2)] for _ in range(row1)]#initialize resultant matrix with all zeroes

    #Check for incompatible dimensions=> matrix multiplication can only happen if row2=col1
    if row2!=col1:
        raise ValueError("Incompatible Dimensions")
        return 0

    # Perform matrix multiplication
    for i in range(row1):  # iterate over rows of matrix1
        for j in range(col2): # iterate over columns of matrix2
            result_matrix[i][j]=0
            for k in range(col1): # iterate over common dimension
                if not isinstance(matrix1[i][k],(int,float)) or not isinstance(matrix2[k][j],(int,float)):# Check for invalid datatype (only integers allowed)
                    raise TypeError("Incorrect Datatype")
                else: # Multiply and add to the running sum
                    result_matrix[i][j]=result_matrix[i][j]+(matrix1[i][k]*matrix2[k][j])
    return result_matrix #return the computed resultant matrix
    
    # Placeholder implementation that always raises NotImplementedError
    raise NotImplementedError("Matrix multiplication function not implemented")

