
import java.util.Random;

public class MatrixMultiplication {

	public static void main(String[] args) {
		int matrixA[][];
		int matrixB[][];
		int resultMatrix[][];
		int size = 2;
		
		long cMMStart, dcMMStart, sMMStart;
		long cMMEnd, dcMMEnd, sMMEnd;
		
		// generate matrices size up to 512
		while (size <= 512) {
			cMMEnd = dcMMEnd = sMMEnd = 0;
			
			// generated two matrices
			matrixA = generatedMatrix(size);
			matrixB = generatedMatrix(size);
			
			// for each set of matrices, run through all three algorithms, 20x times
			for (int i = 0; i < 20; i++) {
				// Classic Multiply Matrix
				cMMStart = System.nanoTime(); 
				resultMatrix = classicMatrixMultiply(matrixA, matrixB);
				cMMEnd += System.nanoTime() - cMMStart;
				 
		        // Divide and Conquer Multiply Matrix
		        dcMMStart = System.nanoTime();
		        resultMatrix = divideAndConquerMM(matrixA, matrixB);
		        dcMMEnd += System.nanoTime() - dcMMStart;
		      
		        // Strassen Multiply Matrix
		        sMMStart = System.nanoTime(); 
		        strassenMM(matrixA, matrixB, resultMatrix, size);
		        sMMEnd += System.nanoTime() - sMMStart; 		        
			}
			
			// print out get the average
			System.out.println("\nAverage RunTime for n = " + size); 
			System.out.println("\tClassic Matrix A: " + cMMEnd/20); 
			System.out.println("\tDivide and Conquer Matrix: " + dcMMEnd/20); 
			System.out.println("\tstrassen Matrix: " + sMMEnd/20); 
			size = size * 2;
		}
			
		
	}
	
	// Strassen's Matrix Multiplication
	public static void strassenMM(int[][] matrixA, int[][] matrixB, int[][] matrixC, int n) {
		int a11[][], a12[][], a21[][], a22[][];
		int b11[][], b12[][], b21[][], b22[][];
		int c11[][], c12[][], c21[][], c22[][];
		
		// if base case, 2x2, multiply matrices
		if (n == 2) {
			matrixC[0][0] = (matrixA[0][0] * matrixB[0][0]) + (matrixA[0][1] * matrixB[1][0]);
			matrixC[0][1] = (matrixA[0][0] * matrixB[0][1]) + (matrixA[0][1] * matrixB[1][1]);
			matrixC[1][0] = (matrixA[1][0] * matrixB[0][0]) + (matrixA[1][1] * matrixB[1][0]);
			matrixC[1][1] = (matrixA[1][0] * matrixB[0][1]) + (matrixA[1][1] * matrixB[1][1]);
		} else {
			n = n/2;

			// otherwise, split each matrix into 4 equal matrices
			a11 = splitMatrix(matrixA, 0, 0);
			a12 = splitMatrix(matrixA, 0, n);
			a21 = splitMatrix(matrixA, n, 0);
			a22 = splitMatrix(matrixA, n, n);

			b11 = splitMatrix(matrixB, 0, 0);
			b12 = splitMatrix(matrixB, 0, n);
			b21 = splitMatrix(matrixB, n, 0);
			b22 = splitMatrix(matrixB, n, n);

			// create new matrices
			int[][] P = new int[n][n];
			int[][] Q = new int[n][n];
			int[][] R = new int[n][n];
			int[][] S = new int[n][n];
			int[][] T = new int[n][n];
			int[][] U = new int[n][n];
			int[][] V = new int[n][n];

			// P = (A11 + A22)(B11 + B22)
			strassenMM(addMatrix(a11, a22), addMatrix(b11, b22), P, n);
			// Q = (A21 + A22)B11
			strassenMM(addMatrix(a21, a22), b11, Q, n);
			// R = A11(B12 - B22)
			strassenMM(a11, subtractMatrix(b12, b22), R, n);
			// S = A22(B21 - B11)
			strassenMM(a22, subtractMatrix(b21, b11), S, n);
			// T = (A11 + A12)B22
			strassenMM(addMatrix(a11, a12), b22, T, n);
			// U = (A21 - A11)(B11 + B12)
			strassenMM(subtractMatrix(a21, a11), addMatrix(b11, b12), U, n);
			// V = (A12 - A22)(B21 + B22)
			strassenMM(subtractMatrix(a12, a22), addMatrix(b21, b22), V, n);

			// C11 = P + S - T + V
			c11 = addMatrix( subtractMatrix( addMatrix(P, S), T), V);
			// C12 = R + T
			c12 = addMatrix(R, T);
			// C21 = Q + S
			c21 = addMatrix(Q, S);
			// C22 = P + R - Q + U
			c22 = addMatrix( subtractMatrix( addMatrix(P, R), Q), U);

			// merge 4 matrices into 1
			mergeMatrix(c11, matrixC, 0, 0);
			mergeMatrix(c12, matrixC, 0, n);
			mergeMatrix(c21, matrixC, n, 0);
			mergeMatrix(c22, matrixC, n, n);
		}
	}
	
	// Divide and Conquer Matrix Multiplication
	public static int[][] divideAndConquerMM(int[][] matrixA, int[][] matrixB) {
		int n = matrixA.length;
		int[][] matrixC = new int[n][n];
		
		int a11[][], a12[][], a21[][], a22[][];
		int b11[][], b12[][], b21[][], b22[][];
		int c11[][], c12[][], c21[][], c22[][];
		
		// if base case, 1x1, multiply two numbers
		if (n == 1) {
			matrixC[0][0] = matrixA[0][0] * matrixB[0][0];
		} else {
			n = n/2;

			// otherwise, split each matrix into 4 equal matrices
			a11 = splitMatrix(matrixA, 0, 0);
			a12 = splitMatrix(matrixA, 0, n);
			a21 = splitMatrix(matrixA, n, 0);
			a22 = splitMatrix(matrixA, n, n);

			b11 = splitMatrix(matrixB, 0, 0);
			b12 = splitMatrix(matrixB, 0, n);
			b21 = splitMatrix(matrixB, n, 0);
			b22 = splitMatrix(matrixB, n, n);
			
			// C11 = (A11 * B11) + (A12 + B21)
			c11 = addMatrix(divideAndConquerMM(a11, b11), divideAndConquerMM(a12, b21));
			// C12 = (A11 * B12) + (A12 + B22)
			c12 = addMatrix(divideAndConquerMM(a11, b12), divideAndConquerMM(a12, b22));
			// C21 = (A21 * B11) + (A22 + B21)
			c21 = addMatrix(divideAndConquerMM(a21, b11), divideAndConquerMM(a22, b21));
			// C22 = (A21 * B12) + (A22 + B22)
			c22 = addMatrix(divideAndConquerMM(a21, b12), divideAndConquerMM(a22, b22));

			// merge 4 matrices into 1
			mergeMatrix(c11, matrixC, 0, 0);
			mergeMatrix(c12, matrixC, 0, n);
			mergeMatrix(c21, matrixC, n, 0);
			mergeMatrix(c22, matrixC, n, n);
		}
		return matrixC;
	}

	// Classical Matrix Multiplication
	private static int [][] classicMatrixMultiply(int [][]matrixA, int[][]matrixB){
		int size = matrixA.length;
		int matrixC[][] = new int[size][size]; 
		
		for (int i = 0; i < size; i++) { 
	        for (int j = 0; j < size; j++) { 
	            for (int k = 0; k < size; k++) 
	                matrixC[i][j] += matrixA[i][k] * matrixB[k][j]; 
	        } 
	    } 
		return matrixC;
	}
	// function that split one matrix into 4 matrices
	private static int[][] splitMatrix(int[][] initialMatrix, int x, int m) {
		int size = initialMatrix.length/2;
	    int[][] newMatrix = new int[size][size];
	        
		for (int i = 0; i < size; i++) {
			int y = m;
			for (int j = 0; j < size; j++) {
				newMatrix[i][j] = initialMatrix[x][y++];
			}
			x++;
		}
		return newMatrix;
	}
	
	// function that marge 4 matrices into 1
	private static void mergeMatrix(int[][] initialMatrix, int[][] newMatrix, int x, int m) {

		for (int i = 0; i < initialMatrix.length; i++) {
			int y = m;
			for (int j = 0; j < initialMatrix.length; j++) {
				newMatrix[x][y++] = initialMatrix[i][j];
			}
			x++;
		}
	}
	
	// function matrices subtraction
	private static int[][] subtractMatrix(int [][]matrixA, int[][]matrixB) {
		int size = matrixA.length;
		int[][] matrixC = new int[size][size];
		
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				matrixC[i][j] = matrixA[i][j] - matrixB[i][j];
			}
		}
		return matrixC;
	}
	
	// function matrices addition
	private static int[][] addMatrix(int [][]matrixA, int[][]matrixB) {
		int size = matrixA.length;
		int[][] matrixC = new int[size][size];

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				matrixC[i][j] = matrixA[i][j] + matrixB[i][j];
			}
		}
		return matrixC;
	}
	
	// generate a matrix with input size of random number
	private static int [][] generatedMatrix(int size)	{
		Random rand=new Random();
		int matrix[][] = new int[size][size]; 
		
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				// generate random number to matrix index
				matrix [i][j]= rand.nextInt(100);
		
		return matrix;	
	}
	
	// print out matrix (for debugging purpose)
	private static void printMatrix(int [][]matrix)	{
		
		for(int i = 0; i < matrix.length; i++)	{
		     for(int j = 0; j < matrix.length; j++)	{	
		         System.out.print(matrix[i][j]+"\t");  
		     }
		     System.out.print("\n");
		}
	}
	
}
