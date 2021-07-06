package edu.rutgers.sakai.java.util;

import java.util.Arrays;

/**
 * Class to convert between bit-mapped {@code byte[]} and {@code boolean[]}.
 * 
 * https://sakai.rutgers.edu/wiki/site/e07619c5-a492-4ebe-8771-179dfe450ae4/bit-to-boolean%20conversion.html
 * 
 * @author Robert Moore
 */
public class BitToBoolean {
	/**
	 * Converts a {@code byte[]} to a {@code boolean[]}. It is assumed that the
	 * values are in most-significant-bit first order. Meaning that most
	 * significant bit of the 0th byte of {@code bits} is the first boolean
	 * value.
	 * 
	 * @param bits
	 *            a binary array of boolean values stored as a {@code byte[]}.
	 * @param significantBits
	 *            the number of important bits in the {@code byte[]}, and
	 *            therefore the length of the returned {@code boolean[]}
	 * @return a {@code boolean[]} containing the same boolean values as the
	 *         {@code byte[]}
	 */
	public static boolean[] convert(byte[] bits, int significantBits) {
		boolean[] retVal = new boolean[significantBits];
		int boolIndex = 0;
		for (int byteIndex = 0; byteIndex < bits.length; ++byteIndex) {
			for (int bitIndex = 7; bitIndex >= 0; --bitIndex) {
				if (boolIndex >= significantBits) {
					// Bad to return within a loop, but it's the easiest way
					return retVal;
				}

				retVal[boolIndex++] = (bits[byteIndex] >> bitIndex & 0x01) == 1 ? true
						: false;
			}
		}
		return retVal;
	}

	/**
	 * Converts a {@code byte[]} to a {@code boolean[]}. It is assumed that the
	 * values are in most-significant-bit first order. Meaning that most
	 * significant bit of the 0th byte of {@code bits} is the first boolean
	 * value.
	 * 
	 * @param bits
	 *            a binary array of boolean values stored as a {@code byte[]}.
	 * @return a {@code boolean[]} containing the same boolean values as the
	 *         {@code byte[]}
	 */
	public static boolean[] convert(byte[] bits) {
		return BitToBoolean.convert(bits, bits.length * 8);
	}

	/**
	 * Converts an {@code boolean[]} to a {@code byte[]} where each bit of the
	 * {@code byte[]} contains a 1 bit for a {@code true} value, and a 0 bit for
	 * a {@code false} value. The {@code byte[]} will contain the 0th index
	 * {@code boolean} value in the most significant bit of the 0th byte.
	 * 
	 * @param bools
	 *            an array of boolean values
	 * @return a {@code byte[]} containing the boolean values of {@code bools}
	 *         as bits.
	 */
	public static byte[] convert(boolean[] bools, boolean reverseOrder) {
		int length = bools.length / 8;
		int mod = bools.length % 8;
		if(mod != 0){
			++length;
		}
		
		int bitIndexFactor = 1;
		int bitIndexStart = 0;
		if(reverseOrder) {
			bitIndexFactor = -1;
			bitIndexStart = 7;
		}
		byte[] retVal = new byte[length];
		int boolIndex = 0;
		for (int byteIndex = 0; byteIndex < retVal.length; ++byteIndex) {
			for (int bitIndex = 7; bitIndex >= 0; --bitIndex) {
				// Another bad idea
				if (boolIndex >= bools.length) {
					return retVal;
				}
				if (bools[boolIndex++]) {
					retVal[byteIndex] |= (byte) (1 << 
							(bitIndexStart + bitIndex * bitIndexFactor));
				}
			}
		}

		return retVal;
	}

	/**
	 * A method for testing the conversion between the {@code byte[]} and {@code boolean[]} representations.
	 * @param args ignored.
	 */
	public static void main(String[] args) {
		byte[] testBytes = new byte[] { (byte) 0xF8, 0x4A };

		System.out.println(Arrays.toString(convert(testBytes, 3)));
		
		boolean[] testBools = new boolean[] {true, false, true, false, false, true, false,true,true};
		byte[] convertedBools = convert(testBools, false);
		for(byte b : convertedBools){
			System.out.print(Integer.toHexString(b&0xFF) + ", ");
		}
	}
}