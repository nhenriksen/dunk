import numpy as np
import sys
import os
from datetime import datetime
import shutil
import subprocess as sp
from paprika import amber
from paprika.analysis import *
import logging as log
log.basicConfig(format='%(levelname)s:%(asctime)s: %(message)s')

class Calculation(object):
    """
    Peforms a hydration free energy calculation for a small molecule in AMBER.

    Order of operations for 3 step alchemical transformation:

    1. Solvated System
          --Decharge Solute--> 
              Solvated System (Decharged Solute)

    2. Solvated System (Decharged Solute)
          --Decouple Solute-->
              Gas Phase System (Decharged Solute)

    3. Gas Phase System (Decharged Solute)
          --Recharge Solute-->
              Gas Phase System

    """


    def __init__(self):
        """
        Class Variables
        ---------------
        residue_name : str
            Residue name of solute, as provided in the mol2 file. Default: MOL
        mol2 : str
            The mol2 file name. Currently only supports single residue molecules.
            Default: mol.mol2
        frcmod : str
            The frcmod file name. Default: mol.frcmod
        water_box : str
            The tleap water box (eg TIP3PBOX, TIP4PEWBOX). Default: TIP3PBOX
        data_dir : str
            Directory path in which to build and run simulation windows. Default: windows
        stash_existing : bool
            If True, rename the data_dir with a date/time string and create a new data_dir.
            If False, continue HFE calculation from last completed step. Default: False
        leap_load_lines : list
        


        """

        # Build files and settings
        self._residue_name = 'MOL'
        self._mol2 = 'mol.mol2'
        self._frcmod = 'mol.frcmod'
        self.water_box = 'TIP3PBOX'
        self.data_dir = './windows'
        self.stash_existing = False

        # Leap loader lines for input files 
        self.leap_load_lines = [
            'source leaprc.gaff',
            'source leaprc.water.tip3p',
            'loadamberparams '+self._frcmod,
            self._residue_name+' = loadmol2 '+self._mol2,
            ]

        # Execution settings
        self.vac_exec = 'pmemd'
        self.solv_exec = 'pmemd'
        self.nstlim = 50000
        self.max_rst_itr = 10
        self.dvdl_thresh = {}
        self.dvdl_thresh['dech'] = 0.15
        self.dvdl_thresh['decp'] = 0.15
        self.dvdl_thresh['rech'] = 0.01

        # Setup lambda divisions for each phase
        self.lambdas = {}
        self.lambdas['dech'] = np.array("0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0".split(), np.float64)
        self.lambdas['decp'] = np.array("0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0".split(), np.float64)
        self.lambdas['rech'] = np.array("0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0".split(), np.float64)

        # Stuff we'll collect
        self.mden_files = {}
        self.dvdl_avgs = {}
        self.dvdl_sems = {}
        self.dvdl_totn = {} # Total number of data points
        self.dvdl_blkn = {} # Number of data points for block uncertainty calculation
        self.spline_x = {}
        self.spline_y = {}
        self.fe_avgs = {}
        self.fe_sems = {}

    
    # Refresh leap_load_lines if mol2 or frcmod changes
    @property
    def residue_name(self):
        return self._residue_name
    @residue_name.setter
    def residue_name(self, new_residue_name):
        self._residue_name = new_residue_name
        self.leap_load_lines[3] = new_residue_name+' = loadmol2 '+os.path.basename(self._mol2)

    @property
    def mol2(self):
        return self._mol2
    @mol2.setter
    def mol2(self, new_mol2):
        self._mol2 = new_mol2
        self.leap_load_lines[3] = self._residue_name+' = loadmol2 '+os.path.basename(new_mol2)

    @property
    def frcmod(self):
        return self._frcmod
    @frcmod.setter
    def frcmod(self, new_frcmod):
        self._frcmod = new_frcmod
        self.leap_load_lines[2] = 'loadamberparams '+os.path.basename(new_frcmod)


    def setup_data_dir(self, stash_existing=True):
        """ Create directory for calculations """
        # Stash existing data_dir if we request it
        if self.stash_existing and os.path.isdir(self.data_dir):
            stash_dir = self.data_dir+"_{:%Y.%m.%d_%H.%M.%S}".format(datetime.now())
            shutil.move(self.data_dir, stash_dir)
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)

    def build_system(self, phase):
        """ Build the system with tleap """

        log.info('Preparing '+phase+' phase with these mol2/frcmod'+self.mol2+' '+self.frcmod)
        # Create a dir to store build files
        dir_name = self.data_dir+'/build_'+phase
        log.info('Build directory: '+dir_name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        # Copy files
        for build_file in [self.mol2, self.frcmod]:
            shutil.copy(build_file, dir_name)
        # Write tleap input file
        with open(dir_name+'/tleap.in', 'w') as f:
            for line in self.leap_load_lines:
                f.write(line + "\n")
            # Build the model ... duplicate moleules for charging steps
            if phase in ['dech', 'rech']:
                f.write("model = combine{{ {0} {0} }}\n".format(self.residue_name))
            elif phase == 'decp':
                f.write("model = combine{{ {0} }}\n".format(self.residue_name))
            else:
                raise Exception('build_system does not recognize phase: '+phase)
            # Solvate for dech and decp
            if phase in ['dech', 'decp']:
                f.write("solvateoct model {} 15.0 iso\n".format(self.water_box))
            # Save files
            f.write("savepdb model {}.pdb\n".format(phase))
            if phase == 'decp':
                f.write("saveamberparm model decp.prmtop decp.rst7\n")
                prmtop_name = 'decp.prmtop'
            else:
                f.write("saveamberparm model init.{0}.prmtop init.{0}.rst7\n".format(phase))
                prmtop_name = 'init.'+phase+'.prmtop'
            f.write("quit\n")
        # Execute tleap
        exec_list = ['tleap','-s','-f','tleap.in']
        with open(dir_name+'/tleap.out', 'w') as f:
            sp.call(exec_list, cwd=dir_name, stdout=f, stderr=sp.STDOUT)
        if not os.path.isfile(dir_name+'/'+prmtop_name):
            raise Exception('tleap was unable to build the prmtop: '+dir_name+'/'+prmtop_name)
        if phase in ['dech', 'rech']:
            # Write parmed input file
            with open(dir_name+'/parmed.in', 'w') as f:
                f.write("""\
parm init.{0}.prmtop
loadcoordinates init.{0}.rst7
timerge :1 :2 :1 :2
parmout {0}.prmtop
writecoordinates {0}.rst7
go
                """.format(phase))
            # Execute parmed
            exec_list = ['parmed', '-O', '-i', 'parmed.in']
            with open(dir_name+'/parmed.out', 'w') as f:
                sp.call(exec_list, cwd=dir_name, stdout=f, stderr=sp.STDOUT)
            if not os.path.isfile(dir_name+'/'+phase+'.prmtop'):
                raise Exception('parmed was unable to prepare '+dir_name+'/'+phase+'.prmtop')

    def run_windows(self, phase):
        """ Setup windows for a TI calculation """

        # Check phase matches expectations
        if not phase in ['dech', 'decp', 'rech']:
            raise Exception('phase must be one of the following: dech, decp, rech')

        log.info('Running '+phase+' phase ....')

        # Setup Simulation
        sim = amber.Simulation()
        if phase == 'rech':
            sim.executable = self.vac_exec
        else:
            sim.executable = self.solv_exec
        sim.disang = None
        sim.topology = phase+'.prmtop'
        sim.cntrl['nmropt'] = 0
        sim.cntrl.pop('pencut', None)
        sim.cntrl['icfe'] = 1
        sim.cntrl['clambda'] = 0.0
        if phase == 'decp':
            sim.cntrl['ntmin'] = 2
            sim.cntrl['ifsc'] = 1
            sim.cntrl['timask1'] = "':1'"
            sim.cntrl['timask2'] = "''"
            sim.cntrl['scmask1'] = "':1'"
            sim.cntrl['scmask2'] = "''"
        else:
            sim.cntrl['ifsc'] = 0
            sim.cntrl['timask1'] = "':1'"
            sim.cntrl['timask2'] = "':2'"
        if phase in ['decp', 'rech']:
            sim.cntrl['crgmask'] = "':1'"
        if phase == 'dech':
            sim.cntrl['crgmask'] = "':2'"

        # Iterate over the restart level
        for itr in range(self.max_rst_itr):
            log.info('Beginning Iteration {:.0f} -------------'.format(itr))
            new_data = False
            prev = "{:03.0f}".format(itr)
            curr = "{:03.0f}".format(itr+1)
            # Iterate over each lambda in this phase
            for l,win_val in enumerate(self.lambdas[phase]):
                dir_name = self.data_dir+'/'+phase+"_{:7.5f}".format(win_val)
                if itr == 0:
                    # Create a window directory
                    if not os.path.isdir(dir_name):
                        os.makedirs(dir_name)
                    # Copy files
                    files = [phase+'.prmtop', phase+'.rst7']
                    src_dir = self.data_dir+'/build_'+phase+'/'
                    for filename in files:
                        if not os.path.isfile(dir_name+'/'+filename):
                            shutil.copy(src_dir+filename, dir_name)
    
                # Run Simulation
                sim.path = dir_name
                sim.cntrl['clambda'] = win_val
                if itr == 0:
                    # Minimize
                    sim.prefix = 'minimize'
                    sim.inpcrd = phase+'.rst7'
                    if phase == 'rech':
                        sim.config_gb_min()
                        sim.cntrl['igb'] = 6
                    else:
                        sim.config_pbc_min()
                    sim.cntrl['maxcyc'] = 500
                    sim.cntrl['ncyc'] = 100
                    sim.cntrl['ntmin'] = 2
                    if not os.path.isfile(dir_name+'/minimize.rst7'):
                        if self.solv_exec == 'pmemd.cuda':
                            sim.executable = 'pmemd'
                        sim.run(overwrite=True)
                        if phase == 'rech':
                            sim.executable = self.vac_exec
                        else:
                            sim.executable = self.solv_exec
    
                # MD
                sim.prefix = 'md.'+curr
                if phase == 'rech':
                    sim.config_gb_md()
                    sim.cntrl['igb'] = 6
                else:
                    sim.config_pbc_md()
                if phase == 'decp':
                    sim.cntrl['ntf'] = 1
                sim.cntrl['nstlim'] = self.nstlim
                sim.cntrl['dt'] = 0.001
                if itr == 0:
                    sim.inpcrd = 'minimize.rst7'
                else:
                    sim.inpcrd = 'md.'+prev+'.rst7'
                    sim.cntrl['ntx'] = 5
                    sim.cntrl['irest'] = 0
                # Check if this itr is already finished:
                itr_not_finished = True
                if os.path.isfile(dir_name+'/'+sim.output):
                    if not ' TIMINGS ' in open(dir_name+'/'+sim.output, 'r').read():
                        itr_not_finished = False
                # Run itr if we aren't finished and we're still above dvdl_thresh
                if itr_not_finished and itr == 0:
                    sim.run(overwrite=True)
                    new_data = True
                elif itr_not_finished and self.dvdl_sems[phase][l] > self.dvdl_thresh[phase]:
                    sim.run(overwrite=True)
                    new_data = True
                elif itr == 0:
                    log.info('Data already exists in '+dir_name+' for this itr round. Continuing ...')
                    pass
                else:
                    log.info('Data in '+dir_name+' is converged. Skipping ...')
                    pass

            # Get dvdl's and free energy
            if new_data or itr == 0:
                self.get_fe(phase)


    def get_mden_files(self, dir_name):
        """ Collects and stores mden files in a list """

        mden_files = []
        i = 1
        while os.path.isfile(dir_name+"/md.{:03.0f}.mden".format(i)):
            mden_files.append(dir_name+"/md.{:03.0f}.mden".format(i))
            i += 1
        return mden_files

    def get_dvdls(self, file_list):
        """ Collect dvdl values from list of mden files """

        dvdl = []
        for mden in file_list:
            with open(mden, 'r') as f:
                lines = f.readlines()
            for i in range(10,len(lines)):
                if i % 10 == 9:
                    cols = lines[i].rstrip().split()
                    dvdl.append(cols[-1])
        return np.array(dvdl, np.float64)
        

    def get_fe(self, phase):
        """ Compute the free energy for a phase """

        # Check phase matches expectations
        if not phase in ['dech', 'decp', 'rech']:
            raise Exception('phase must be one of the following: dech, decp, rech')

        log.info('Computing free energy via bootstrapping on phase: '+phase)
        self.dvdl_avgs[phase] = np.zeros([self.lambdas[phase].size], np.float64)
        self.dvdl_sems[phase] = np.zeros([self.lambdas[phase].size], np.float64)
        self.mden_files[phase] = []
        self.dvdl_totn[phase] = np.zeros([self.lambdas[phase].size], np.int32)
        self.dvdl_blkn[phase] = np.zeros([self.lambdas[phase].size], np.int32)
        
        # Get dvdl avg/sem for each lambda window
        for i,win_val in enumerate(self.lambdas[phase]):
            dir_name = self.data_dir+'/'+phase+"_{:7.5f}".format(float(win_val))
            # If we haven't populated the list, then append
            if len(self.mden_files[phase]) == i:
                self.mden_files[phase].append(self.get_mden_files(dir_name))
            # Else, rewrite
            else:
                self.mden_files[phase][i] = self.get_mden_files(dir_name)
            data = self.get_dvdls(self.mden_files[phase][i])
            self.dvdl_avgs[phase][i] = np.mean(data)
            self.dvdl_totn[phase][i] = data.size
            nearest_max = get_nearest_max(len(data))
            self.dvdl_blkn[phase][i] = nearest_max
            self.dvdl_sems[phase][i] = get_block_sem(data[:nearest_max])
            log.info("idx:{:4.0f} lambda:{:7.5f} totn:{:8.0f} blkn:{:8.0f} avg:{:8.3f} sem:{:8.3f}".format(i,win_val,self.dvdl_totn[phase][i],nearest_max,self.dvdl_avgs[phase][i],self.dvdl_sems[phase][i]))

        # We're gonna create a spline through our data and then integrate

        # Create an array to store the x-values of a spline. We're gonna
        # do 100 evenly spaced points between each window. There might be
        # a better way to create this thing, but this ensures that I have
        # a spline point exactly at each window
        # (Could the even spacing somehow bias the calculations in a bad way, if
        #  the windows themselves are not evenly spaced from 0->1?)
        self.spline_x[phase] = np.zeros( [0], np.float64 ) # array for x dimension spline points
        for i in range(self.lambdas[phase].size-1):
            tmp_x = np.linspace(self.lambdas[phase][i], self.lambdas[phase][i+1], num=100, endpoint=False)
            self.spline_x[phase] = np.append(self.spline_x[phase], tmp_x)
        self.spline_y[phase] = interpolate(self.lambdas[phase], self.dvdl_avgs[phase], self.spline_x[phase])

        # We're gonna bootstrap a bunch of fe values. Create storage arrays
        boot_cycs = 10000
        fe_boot = np.zeros([boot_cycs], np.float64)
        # We'll precompute all samples of dvdls based on our dvdl_avg/sem
        boot_dvdls = np.zeros([self.lambdas[phase].size, boot_cycs], np.float64)
        for i in range(self.lambdas[phase].size):
            boot_dvdls[i][:] = np.random.normal(self.dvdl_avgs[phase][i],self.dvdl_sems[phase][i],boot_cycs)
        for cyc in range(boot_cycs):
            boot_spline_y = interpolate(self.lambdas[phase], boot_dvdls[:,cyc], self.spline_x[phase])
            fe_boot[cyc] = np.trapz(boot_spline_y, self.spline_x[phase])

        # Compute the free energy and uncertainty
        self.fe_avgs[phase] = np.mean(fe_boot)
        self.fe_sems[phase] = np.std(fe_boot)

        log.info("{} free energy: {:8.3f} ({:8.3f})".format(phase,self.fe_avgs[phase],self.fe_sems[phase]))

        


        
            
            






def interpolate(x, y, x_new, axis=-1, out=None):
    # Copyright (c) 2007-2015, Christoph Gohlke
    # Copyright (c) 2007-2015, The Regents of the University of California
    # Produced at the Laboratory for Fluorescence Dynamics
    # All rights reserved.
    #
    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions are met:
    #
    # * Redistributions of source code must retain the above copyright
    #   notice, this list of conditions and the following disclaimer.
    # * Redistributions in binary form must reproduce the above copyright
    #   notice, this list of conditions and the following disclaimer in the
    #   documentation and/or other materials provided with the distribution.
    # * Neither the name of the copyright holders nor the names of any
    #   contributors may be used to endorse or promote products derived
    #   from this software without specific prior written permission.
    #
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    # ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    # POSSIBILITY OF SUCH DAMAGE.
    x = np.array(x, dtype=np.float64, copy=True)
    y = np.array(y, dtype=np.float64, copy=True)
    xi = np.array(x_new, dtype=np.float64, copy=True)
    if axis != -1 or out is not None or y.ndim != 1:
        raise NotImplementedError("implemented in C extension module")
    if x.ndim != 1 or xi.ndim != 1:
        raise ValueError("x-arrays must be one dimensional")
    n = len(x)
    if n < 3:
        raise ValueError("array too small")
    if n != y.shape[axis]:
        raise ValueError("size of x-array must match data shape")
    dx = np.diff(x)
    if any(dx <= 0.0):
        raise ValueError("x-axis not valid")
    if any(xi < x[0]) or any(xi > x[-1]):
        raise ValueError("interpolation x-axis out of bounds")
    m = np.diff(y) / dx
    mm = 2.0 * m[0] - m[1]
    mmm = 2.0 * mm - m[0]
    mp = 2.0 * m[n - 2] - m[n - 3]
    mpp = 2.0 * mp - m[n - 2]
    m1 = np.concatenate(([mmm], [mm], m, [mp], [mpp]))
    dm = np.abs(np.diff(m1))
    f1 = dm[2:n + 2]
    f2 = dm[0:n]
    f12 = f1 + f2
    ids = np.nonzero(f12 > 1e-9 * np.max(f12))[0]
    b = m1[1:n + 1]
    b[ids] = (f1[ids] * m1[ids + 1] + f2[ids] * m1[ids + 2]) / f12[ids]
    c = (3.0 * m - 2.0 * b[0:n - 1] - b[1:n]) / dx
    d = (b[0:n - 1] + b[1:n] - 2.0 * m) / dx ** 2
    bins = np.digitize(xi, x)
    bins = np.minimum(bins, n - 1) - 1
    bb = bins[0:len(xi)]
    wj = xi - x[bb]
    return ((wj * d[bb] + c[bb]) * wj + b[bb]) * wj + y[bb]


