package com.github.neuralnetworks.tensor;

import java.util.Arrays;


/**
 * Simple matrix representation with one-dimensional array. This is required,
 * because Aparapi supports only one-dim arrays (otherwise the execution is
 * transferred to the cpu)
 */
public class Matrix extends Tensor {

    private static final long serialVersionUID = 1L;

    public Matrix(Tensor parent, int[][] dimensionsLimit) {
	super(parent, dimensionsLimit);
    }

    public Matrix(int startOffset, float[] elements, int[] globalDimensions, int[][] globalDimensionsLimit) {
	super(startOffset, elements, globalDimensions, globalDimensionsLimit);
    }

    public int getColumns() {
	return getDimensions()[1];
    }

    public int getColumnElementsDistance() {
	return getDimensionElementsDistance(getDimensionGlobalIndex(1));
    }
    
    public void setColumn(int j, float[] column) {
	if (j < 0 || j > getColumns() || column.length != getRows()) {
	    throw new IllegalArgumentException();
	}
	
	for (int i = 0; i < getRows(); i++) {
	    set(column[i], i, j);
	}
    }
    
    public float[] getColumn(int j) {
	float[] column = new float[getRows()];
	for (int i = 0; i < getRows(); i++) {
	    column[i] = get(i, j);
	}
	return column;
    }

    public int getRows() {
	return getDimensions()[0];
    }

    public int getRowElementsDistance() {
	return getDimensionElementsDistance(getDimensionGlobalIndex(0));
    }

    public void setRow(int i, float[] row) {
	if (i < 0 || i > getRows() || row.length != getColumns()) {
	    throw new IllegalArgumentException();
	}
	
	for (int j = 0; j < getColumns(); j++) {
	    set(row[j], i, j);
	}
    }
    
    public float[] getRow(int i) {
	float[] row = new float[getColumns()];
	for (int j = 0; j < getColumns(); j++) {
	    row[j] = get(i, j);
	}
	return row;
    }
    
    public Matrix transpose() {
	Matrix transposed = TensorFactory.tensor(this.getColumns(), this.getRows());
	for (int i = 0; i < getRows(); i++) {
	    transposed.setColumn(i, this.getRow(i));
	}
	return transposed;
    }
    
    @Override
    public String toString() {
	return "Matrix" + Arrays.toString(getDimensions()) + ": " + Arrays.toString(getElements());
    }
}
