from dunk import hfe
import pytest
import os
import shutil
from glob import glob
import filecmp
import numpy as np

def test_setup_data_dir():

    # Remove previous output, create fresh output
    if os.path.exists('output'):
        shutil.rmtree('output')
    os.makedirs('output')

    # Check creation of data_dir
    calc = hfe.Calculation()
    calc.data_dir = 'output/windows'
    calc.setup_data_dir()
    assert os.path.exists('./output/windows')

    # Check stashing
    calc = hfe.Calculation()
    calc.stash_existing = True
    calc.data_dir = 'output/windows'
    calc.setup_data_dir()
    dirs = glob('output/windows*')
    assert len(dirs) == 2

    # Remove output
    shutil.rmtree('output')

def test_build_system():

    # Remove previous output, create fresh output
    if os.path.exists('output'):
        shutil.rmtree('output')
    os.makedirs('output')

    # Check prmtops/rst7 files
    calc = hfe.Calculation()
    calc.residue_name = 'THF'
    calc.mol2 = 'input/thf.mol2'
    calc.frcmod = 'input/thf.frcmod'
    calc.data_dir = 'output/windows'
    calc.setup_data_dir()
    for phase in ['decharge', 'decouple', 'recharge']:
        # Build files
        calc.build_system(phase)
        # Compare files
        for filetype in ['.prmtop', '.rst7']:
            with open('output/windows/build_'+phase+'/'+phase+filetype, 'r') as f:
                # We're gonna skip the first line because in the prmtop it has a date
                test_file = f.readlines()[1:]
            with open('reference/thf.'+phase+filetype, 'r') as f:
                ref_file = f.readlines()[1:]
            assert test_file == ref_file

    # Remove output
    shutil.rmtree('output')

def test_run_windows_no_pmemd():

    # Remove previous output, create fresh output
    if os.path.exists('output'):
        shutil.rmtree('output')
    os.makedirs('output')

    # Reference values
    ref_dvdl_avgs = {
        'decharge': np.array([-4.75661423, -4.75736718, -4.75811992]),
        'decouple': np.array([ 6.73452912,  6.51572821,  6.31032105]),
        'recharge': np.array([ 0.16487367,  0.16482078,  0.16476788]),
    }

    # Setup calculation
    calc = hfe.Calculation()
    calc.residue_name = 'THF'
    calc.mol2 = 'input/thf.mol2'
    calc.frcmod = 'input/thf.frcmod'
    calc.data_dir = 'output/windows'
    calc.vac_exec = 'sander'
    calc.solv_exec = 'sander'
    calc.no_pmemd = True
    calc.maxcyc = 1
    calc.nstlim = 2
    calc.ntwe = 1
    calc.ntp = 0
    calc.ig = 39383
    calc.max_itr = 1
    calc.lambdas['decharge'] = np.array("0.0 0.5 1.0".split(), np.float64)
    calc.lambdas['decouple'] = np.array("0.0 0.5 1.0".split(), np.float64)
    calc.lambdas['recharge'] = np.array("0.0 0.5 1.0".split(), np.float64)
    calc.setup_data_dir()
    for phase in ['decharge', 'decouple', 'recharge']:
        # Build files
        calc.build_system(phase)
        # Run windows
        calc.run_windows(phase)
        # Compare
        for filetype in ['0.00000_md.001.out', '0.50000_md.001.out', '1.00000_md.001.out']:
            cols = filetype.split('_')
            with open('output/windows/'+phase+'_'+cols[0]+'/'+cols[1], 'r') as f:
                # We're gonna skip the first 12 lines because they environment settings
                # at runtime that will fail the test.
                test_file = f.readlines()[12:]
            with open('reference/thf_'+phase+'_'+filetype, 'r') as f:
                ref_file = f.readlines()[12:]
            assert test_file == ref_file

    # Remove output
    shutil.rmtree('output')


@pytest.mark.slow
def test_run_windows_with_pmemd():

    # Remove previous output, create fresh output
    if os.path.exists('output'):
        shutil.rmtree('output')
    os.makedirs('output')

    # Reference values
    ref_dvdl_avgs = {
        'decharge': np.array([-4.75661423, -4.75736718, -4.75811992]),
        'decouple': np.array([ 6.73452912,  6.51572821,  6.31032105]),
        'recharge': np.array([ 0.16487367,  0.16482078,  0.16476788]),
    }

    # Setup calculation
    calc = hfe.Calculation()
    calc.residue_name = 'THF'
    calc.mol2 = 'input/thf.mol2'
    calc.frcmod = 'input/thf.frcmod'
    calc.data_dir = 'output/windows'
    calc.vac_exec = 'sander'
    calc.solv_exec = 'sander'
    calc.maxcyc = 1
    calc.nstlim = 2
    calc.ntwe = 1
    calc.ntp = 0
    calc.ig = 39383
    calc.max_itr = 1
    calc.lambdas['decharge'] = np.array("0.0 0.5 1.0".split(), np.float64)
    calc.lambdas['decouple'] = np.array("0.0 0.5 1.0".split(), np.float64)
    calc.lambdas['recharge'] = np.array("0.0 0.5 1.0".split(), np.float64)
    calc.setup_data_dir()
    for phase in ['decharge', 'decouple', 'recharge']:
        # Build files
        calc.build_system(phase)
        # Run windows
        calc.run_windows(phase)
        # Compare
        assert np.allclose(calc.dvdl_avgs[phase], ref_dvdl_avgs[phase])

    # Remove output
    shutil.rmtree('output')

def test_get_fe():

    # Setup calculation to look for data already available.
    # The test data is for methane.
    calc = hfe.Calculation()
    calc.data_dir = 'reference/mden_data'
    # The decouple data has some extra windows than default
    calc.lambdas['decouple'] = np.array("0.0 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0".split(), np.float64)
    # Sum over the three phases for avg and sem
    avg_sum = 0.0
    sem_sum = 0.0
    for phase in ['decharge', 'decouple', 'recharge']:
        calc.get_fe(phase)
        avg_sum += calc.fe_avgs[phase]
        sem_sum += calc.fe_sems[phase]**2

    # Compare with reference. These are bootstrapped values so we'll
    # need to increase the tolerance for np.allclose
    # Compare the mean hydration free energy with reference value
    assert np.allclose([-1.0*avg_sum], [2.471], atol=2.0e-3)
    # Compare the computed SEM with reference value
    assert np.allclose([np.sqrt(sem_sum)], [0.051], atol=2.0e-3)




