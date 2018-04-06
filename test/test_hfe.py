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

@pytest.mark.slow
def test_run_windows():

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






