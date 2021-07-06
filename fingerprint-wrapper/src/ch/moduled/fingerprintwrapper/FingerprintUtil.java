package ch.moduled.fingerprintwrapper;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.Console;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Vector;
import java.util.Base64;
import java.util.Base64.Encoder;
import java.util.BitSet;
import java.util.List;
import java.util.Stack;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.concurrent.atomic.AtomicInteger;

import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.inchi.InChIGenerator;
import org.openscience.cdk.inchi.InChIGeneratorFactory;
import org.openscience.cdk.interfaces.IAtomContainer;
///import org.openscience.cdk.qsar.descriptors.molecular.XLogPDescriptor;
//import org.openscience.cdk.qsar.result.DoubleResult;
import org.openscience.cdk.silent.SilentChemObjectBuilder;
import org.openscience.cdk.smiles.SmiFlavor;
import org.openscience.cdk.smiles.SmilesGenerator;
import org.openscience.cdk.smiles.SmilesParser;
import org.openscience.cdk.tools.CDKHydrogenAdder;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

import de.unijena.bioinf.ChemistryBase.chem.SmilesU;
import de.unijena.bioinf.ChemistryBase.fp.ArrayFingerprint;
import de.unijena.bioinf.ChemistryBase.fp.BooleanFingerprint;
import de.unijena.bioinf.ChemistryBase.fp.CdkFingerprintVersion;
import de.unijena.bioinf.ChemistryBase.fp.Fingerprint;
import de.unijena.bioinf.ChemistryBase.fp.FingerprintVersion;
//import de.unijena.bioinf.ChemistryBase.fp.Fingerprint;
import de.unijena.bioinf.fingerid.Fingerprinter;
import de.unijena.bioinf.fingerid.fingerprints.FixedFingerprinter;
import de.unijena.bioinf.fingerid.fingerprints.FixedFingerprinter.FixedFingerprinterInstance;
import edu.rutgers.sakai.java.util.BitToBoolean;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import me.tongfei.progressbar.ProgressBarStyle;


public class FingerprintUtil{
	
	// MST: fingerprinting extension
	protected CdkFingerprintVersion version; // = CdkFingerprintVersion.getDefault();
    protected InChIGeneratorFactory inChIGeneratorFactory;
    protected SmilesGenerator smilesGen, smilesGenCanonical;
    protected SmilesParser smilesParser;
    protected Encoder b64;
    protected Stack<FixedFingerprinter> fixedFingerprinters;
	
	public static FingerprintUtil instance = new FingerprintUtil();
	
	
	public FingerprintUtil() 
	{
		try {
			this.version = CdkFingerprintVersion.getDefault();
			fixedFingerprinters = new Stack<>();
			
			//this.fingerprinter = Fingerprinter.getFor(version);
			this.inChIGeneratorFactory = InChIGeneratorFactory.getInstance();
            smilesGen = new SmilesGenerator(
            		SmiFlavor.Generic | 
            		SmiFlavor.UseAromaticSymbols);
            smilesGenCanonical = new SmilesGenerator(
            		SmiFlavor.Canonical |
            		SmiFlavor.UseAromaticSymbols);
            b64 = Base64.getEncoder(); 
		}
		catch(CDKException e)
		{
			// fail forever
		}

	}
	
	public int makeFingerprinters(int n) throws Exception
	{
		for(int i=0;i<n;i++) {
			fixedFingerprinters.push(new FixedFingerprinter(version));
		}
		return fixedFingerprinters.size();
	}
	
    
    // 5.9.18 Michele Stravs:
    // Most of this code is copied from SIRIUS:
    // de.unijena.bioinf.fingerid.db.CustomDatabase.Importer.computeCompound(IAtomContainer, String, String)
    // Adapted to maximal simplicity / least dependencies. 
	
    // 30.4.20 Michele Stravs:
    /*
     * Currently used in fingerprinter_oss de.unijena.bioinf.fingerid.fingerprints
     * FixedFingerprint.class
     *  public ArrayFingerprint computeFingerprintFromSMILES(String smiles) {
        return new BooleanFingerprint(cdkFingerprintVersion, 
        new FixedFingerprinterInstance(
        	FixedFingerprinter.parseStructureFromStandardizedSMILES(smiles),
        	false).getAsBooleanArray()).asArray();
       }

    public static IAtomContainer parseStructureFromStandardizedSMILES(String smiles) {
        try {
            IAtomContainer mol = new SmilesParser(SilentChemObjectBuilder.getInstance()).parseSmiles(smiles);
            initializeMolecule(mol,false);
            return mol;
        } catch (CDKException e) {
            throw new RuntimeException(e);
        }
    }

    private static void initializeMolecule(IAtomContainer molecule, boolean hotfix) throws CDKException {
        CDKHydrogenAdder adder = CDKHydrogenAdder.getInstance(SilentChemObjectBuilder.getInstance());
        AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(molecule);
        adder.addImplicitHydrogens(molecule);
        if (hotfix) removeStrangeImidoSubstructure(molecule);
    }

     */
    
    protected byte[] getFingerprint(final String smiles) throws Exception {
    	byte[] fpByteArray;
		FixedFingerprinter fingerprinter = fixedFingerprinters.pop();
		try {
	        ArrayFingerprint fpArray = fingerprinter.computeFingerprintFromSMILES(smiles);
	        var ozs = fpArray.toOneZeroString();
	        fpByteArray = getBytesFingerprint(fpArray);
	    
		}
		finally {
			fixedFingerprinters.push(fingerprinter);
		}
		
    	return(fpByteArray);
    }
    
    public byte[] getBytesFingerprint(Fingerprint fingerprint) {
        BooleanFingerprint fpBool = fingerprint.asBooleans();
        boolean[] fpBoolArray = fpBool.toBooleanArray();
        //bfp
        byte[] fpByteArray = BitToBoolean.convert(fpBoolArray, true);
        return fpByteArray;
    }
    
    public String getBase64Fingerprint(Fingerprint fingerprint) {
    	return this.b64.encodeToString(
    			this.getBytesFingerprint(fingerprint));
    }
    
    public String getBase64Fingerprint(byte[] fingerprint) {
    	return this.b64.encodeToString(fingerprint);
    }
    
    
    /*
     * Returns three SMILES: 
     * * text-stripped -> parsed -> canonical export,
     * * text-stripped -> parsed -> aromaticity -> canonical export,
     * * text-stripped -> parsed -> aromaticity -> generic export
     */
    private String[] parseAndNormalizeSmiles(String smiles) throws Exception 
    {
    	
    	String smiles2d = SmilesU.get2DSmilesByTextReplace(smiles);
    	// First: just parsing
    	IAtomContainer molecule = FixedFingerprinter.
    			parseStructureFromStandardizedSMILES(smiles2d);
    	String smilesParsed = this.smilesGenCanonical.create(molecule);

        // Then: Perceive aromaticity the way SIRIUS does it
    	var fixedFpInstance = new FixedFingerprinterInstanceClone(molecule);
    	fixedFpInstance.perceiveAromaticity();
    	molecule = fixedFpInstance.getMolecule();
    	
    	// Then export to SMILES
    	String smilesGeneric = this.smilesGen.create(molecule);
    	String smilesCanonical = this.smilesGenCanonical.create(molecule);
    	
    	return new String[]{smilesParsed, smilesGeneric, smilesCanonical};
    }

    public Vector<Object[]> processSmilesFromPython(String[] smiles, int threads,
    		boolean calcFingerprint, boolean progress) throws Exception {
    	return processSmilesFromPython(smiles, threads, calcFingerprint, true, progress);
    }
    

    // This is just needed to sidestep the "final" requirement on pb in processSmilesFromPython
    protected ProgressBar makeProgressBar(int size, boolean show) {
    	if(!show)
    		return(null);
    	ProgressBarBuilder pbb = new ProgressBarBuilder()
    			.setTaskName("Processing SMILES")
    			.setStyle(ProgressBarStyle.ASCII)
    			.setInitialMax(size);
    	return(pbb.build());
    }
    
    public Vector<Object[]> processSmilesFromPython(String[] smiles, int threads,
    		boolean calcFingerprint, boolean returnB64, boolean progress)
    				throws Exception
    {
    	ThreadPoolExecutor executor = 
    			(ThreadPoolExecutor) Executors.newFixedThreadPool(threads);
    	Vector<Object[]> v = new Vector<>();
    	ArrayList<Future<?>> results = new ArrayList<>();
    	List<String> smilesList = new ArrayList<String>(Arrays.asList(smiles));

    	ProgressBar pb = makeProgressBar(smilesList.size(), progress);
    	try {
    			int id = 0;
    			for(String smiles_: smilesList)
    			{
    				int id_ = id++;
    				results.add(executor.submit( () -> {
        					try{
        						String smiles2 = new String(smiles_);
        						String[] smilesParsed = this.parseAndNormalizeSmiles(smiles2);
        	    				if(calcFingerprint) {
            						byte[] fp = this.getFingerprint(smiles_);
            						if(returnB64) {
                	    				String fpBase64 = this.b64.encodeToString(fp);
                						v.add(new Object[]{
                								id_, 
                								smilesParsed[1],
                								smilesParsed[2],
                								fpBase64});			            							
            						}
            						else {
                						v.add(new Object[]{
                								id_, 
                								smilesParsed[1],
                								smilesParsed[2],
                								fp});			
            							
            						}
        	    				}
        	    				else {
            						v.add(new Object[]{
            								id_,
            								smilesParsed[1],
            								smilesParsed[2]
            										});  
        	    				}
        						//byte[] fp_bytes = fp.toByteArray();
        					}
        					catch(Exception e){
        						if(calcFingerprint)
        							v.add(new Object[] {id_,"", "",null});
        						else
        							v.add(new Object[] {id_, "", ""});
        					}
        					finally {
        						if(pb != null)
        							pb.step();
    	    				}
    				}));
    			}
    	}
    	catch(Exception e) {
    		
    	}
		// Wait for all tasks to finish, then close file
		for(Future<?> f: results)
		{
			f.get();
		}
		if(pb != null)
			pb.close();
		executor.shutdown();
		return(v);

    }
    
    public Object[] getTestFingerprint() throws Exception {
    	String testSmiles = "CN1CC2=C(C=CC3=C2OCO3)C4C1C5=CC6=C(C=C5CC4O)OCO6";
    	return getTestFingerprint(testSmiles);
    }
    
    public Object[] getTestFingerprint(String testSmiles) throws Exception {
    	String[] smilesParsed = this.parseAndNormalizeSmiles(testSmiles);
		var fingerprinter = this.fixedFingerprinters.peek();
        ArrayFingerprint fpArray = fingerprinter.computeFingerprintFromSMILES(smilesParsed[0]);
        return new Object[] {fpArray};
    }
    
    public static void main(String[] args) throws Exception
    {
    	
    	//String test_smiles_ = "CN1CC2=C(C=CC3=C2OCO3)C4C1C5=CC6=C(C=C5CC4O)OCO6";
    	String test_smiles = "CCCCCCCCCCCCCC1=CC(=O)C2=CC=CC=C2N1C";
    	CdkFingerprintVersion version = CdkFingerprintVersion.getDefault();
    	FixedFingerprinter fingerprinter = new FixedFingerprinter(version);
        ArrayFingerprint fpArray = fingerprinter.computeFingerprintFromSMILES(test_smiles);
        var bfp = fpArray.asBooleans();
        return;
    }
    	
    public static void test(String [] args) throws Exception
    {
    	FingerprintUtil fpu_ = FingerprintUtil.instance;
    	//fpu_.makeFingerprinters(4);
    	//var fp = fpu_.processSmilesFromPython(new String[]{test_smiles}, 2, true);
    	
    	
    	int threads = Integer.parseInt(args[0]);
    	String smilesInputFile = args[1];
    	String fpOutputFile = args[2];
    	boolean lineno = false;
    	if(args.length > 3) {
    		int linenoArg = Integer.parseInt(args[3]);
    		if(linenoArg > 0)
    			lineno = true;
    	}
    	FingerprintUtil fpu = FingerprintUtil.instance;
    	fpu.process(smilesInputFile, fpOutputFile, threads, lineno);
    }
    
    public void process(String smilesInputFile, String fpOutputFile, 
    		int threads, boolean printLineno) throws Exception
    {
    	
    	this.makeFingerprinters(threads*2);
    	long t0 = System.currentTimeMillis();
    	long tn = 0;

    	ThreadPoolExecutor executor = 
    			(ThreadPoolExecutor) Executors.newFixedThreadPool(threads);

    	AtomicInteger smilesSuccessCount = new AtomicInteger();
    	AtomicInteger smilesFailureCount = new AtomicInteger();
    	
    	
    	ArrayList<Future<?>> results = new ArrayList<>();
    	
    	try {
    		BufferedWriter w = new BufferedWriter(new FileWriter(fpOutputFile));
    		try(BufferedReader r = new BufferedReader(new FileReader(smilesInputFile))) {
    			int lineno = 1;
    			String line = r.readLine();
    			while(line != null)
    			{
    				String smiles = line;
    				int lineno_ = lineno; 
    				results.add(executor.submit( () -> {
        					try{
        						
        						String[] smilesParsed = this.parseAndNormalizeSmiles(smiles);
            					byte[] fp = this.getFingerprint(smiles);
            					String fpBase64 = this.b64.encodeToString(fp);
        						
        						if(printLineno)
        							w.write(smilesParsed[1] + "\t" 
        									+ smilesParsed[2] + "\t" 
        									+ fpBase64 + "\t" + Integer.toString(lineno_) + "\n");
        						else
        							w.write(smilesParsed[1] + "\t" 
        									+ smilesParsed[2] + "\t" 
        									+ fpBase64 + "\n");
        						long ti = System.currentTimeMillis();
        						//success++;
        						//smiles_success.add(smiles);
        						int success = smilesSuccessCount.incrementAndGet();
        						//int success = smiles_success.size(); 
        						float ops = ((float)(ti - t0)) / success; // /success;
        						if(success % 100 == 0 )
	        						System.out.println("" + success + " processed, "
	        								+ "" + ops + " ms/op ["+smiles +"]");

        					}
        					catch(Exception e)
        					{
        						e.printStackTrace();
        						System.out.println("Failure, ignoring: " + smiles);
        						smilesFailureCount.incrementAndGet();
        						//failure++;
        					}
    					}));
    				line = r.readLine();
    				lineno++;
    			}
    		}
    		catch(Exception e)
    		{
    			System.out.println("Failure");
    		}
    		
    		// Wait for all tasks to finish, then close file
    		for(Future<?> f: results)
    		{
    			f.get();
    		}
    		executor.shutdown();
    		w.close();
    	}
    	catch(Exception e)
    	{
    		e.printStackTrace();
    	}
    	tn = System.currentTimeMillis();
    	System.out.println("Finished, " + smilesSuccessCount.get() + " processed, " 
    			+ smilesFailureCount.get() + " failed in " + (tn-t0)/1000 + "s" );
    }

}
    