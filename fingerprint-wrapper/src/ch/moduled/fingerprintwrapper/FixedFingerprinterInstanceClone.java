package ch.moduled.fingerprintwrapper;

import static org.openscience.cdk.CDKConstants.ISAROMATIC;

import java.util.Set;

import org.openscience.cdk.aromaticity.Aromaticity;
import org.openscience.cdk.aromaticity.ElectronDonation;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.graph.CycleFinder;
import org.openscience.cdk.graph.Cycles;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.silent.SilentChemObjectBuilder;
import org.openscience.cdk.tools.CDKHydrogenAdder;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;



public class FixedFingerprinterInstanceClone {

    protected final IAtomContainer molecule;
    protected boolean implicit;

    protected Set<IBond> cdkAromaticBonds;


    public FixedFingerprinterInstanceClone(IAtomContainer molecule) {
        try {
            this.molecule = molecule;//.clone();
            initialize();
        } catch (CDKException e) {
            throw new RuntimeException(e);
        }
    }

    public void perceiveAromaticity() {
        try {
            if (cdkAromaticBonds==null) {
                final CycleFinder cycles = Cycles.or(Cycles.all(), Cycles.all(6));
                final Aromaticity aromaticity = new Aromaticity(ElectronDonation.daylight(), cycles);
                cdkAromaticBonds = aromaticity.findBonds(molecule);
            }
            // clear existing flags
            molecule.setFlag(ISAROMATIC, false);
            for (IBond bond : molecule.bonds())
                bond.setIsAromatic(false);
            for (IAtom atom : molecule.atoms())
                atom.setIsAromatic(false);

            // set the new flags
            for (final IBond bond : cdkAromaticBonds) {
                bond.setIsAromatic(true);
                bond.getBegin().setIsAromatic(true);
                bond.getEnd().setIsAromatic(true);
            }
            molecule.setFlag(ISAROMATIC, !cdkAromaticBonds.isEmpty());
        } catch (CDKException e) {
            throw new RuntimeException(e);
        }

    }
    
    private void initialize() throws CDKException {
        initializeMolecule(molecule,false);
        this.implicit = true;
    }

    // This is from FixedFingerprinter (not from FixedFingerprinterInstance) 
    private static void initializeMolecule(IAtomContainer molecule, boolean hotfix) throws CDKException {
        CDKHydrogenAdder adder = CDKHydrogenAdder.getInstance(SilentChemObjectBuilder.getInstance());
        AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(molecule);
        adder.addImplicitHydrogens(molecule);
        /*if (hotfix) removeStrangeImidoSubstructure(molecule);*/
    }
    
    public IAtomContainer getMolecule() {
    	return this.molecule;
    }

    
}