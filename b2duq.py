"""
Runs a dimension adaptive stochastic colocation UQ campaign on the blob2d model
from BOUT++.  Should be run with python3 in the same folder as blob2d and a
blob2d input template (found in same github repo as this code).

Dependencies: a blob2d build, easyvvuq-1.2 & xbout-0.3.5.
"""

import easyvvuq as uq
import numpy as np
import scipy.constants as const
import chaospy as cp
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from easyvvuq.actions import CreateRunDirectory, Encode, Decode, ExecuteLocal, Actions
from pprint import pprint

from easyvvuq import OutputType
from xbout import open_boutdataset

class B2dDecoder:
    """
    Custom decoder for blob2d output.

    Parameters
    ----------
    target_filename (str)
        Name of blob2d output file to be decoded.
    ouput_columns (list)
        List of output quantities to considered by the campaign
    output_type (OutputType object)
        Easyvvuq object describing format of data returned by decoder
    """
    
    def __init__(self, target_filename, output_columns):
        self.target_filename = target_filename
        self.output_columns = output_columns
        self.output_type = OutputType('sample')
    
    def peak_reached(self, vels):
        """
        Returns a boolean describing whether the blob CoM velocity has reached a
        peak.
        """
        if max(vels) != vels[-1]: return True
        else: return False
    
    def get_blob_info(self, out_path):    
        """
        Uses xbout to extract the data from blob2d output files and convert to useful quantities.
        
        Parameters
        ----------
        out_path (str)
            Absolute path to the blob2d output files.

        Returns
        -------
        blobInfo (dict)
            Dictionary of quantities which may be called by the campaign.
            Also contains whether the simulation peaked
        """
        
        # Unpack data from blob2d
        ds = open_boutdataset(out_path, info=False)
        ds = ds.squeeze(drop=True)
        dx = ds["dx"].isel(x=0).values
        ds = ds.drop("x")
        ds = ds.assign_coords(x=np.arange(ds.sizes["x"])*dx)
        
        # Obtain blob info from data
        blobInfo = {}
        background_density = 1.0
        ds["delta_n"] = ds["n"] - background_density
        integrated_density = ds.bout.integrate_midpoints("delta_n")
        ds["delta_n*x"] = ds["delta_n"] * ds["x"]
        ds["transpRate"] = ds.bout.integrate_midpoints("delta_n*x")
        ds["CoM_x"] = ds["transpRate"] / integrated_density
        v_x = ds["CoM_x"].differentiate("t")
        
        # Save useful quantities to dictionary
        maxV = float(max(v_x))
        maxX = float(ds["CoM_x"][list(v_x).index(max(v_x))])
        avgTransp = float(np.mean(ds["transpRate"][:(list(v_x).index(max(v_x)))+1]))
        massLoss = float(integrated_density[list(v_x).index(max(v_x))] / integrated_density[0])
        peaked = self.peak_reached(list(v_x))
        blobInfo = {"maxV": maxV, "maxX": maxX, "avgTransp": avgTransp, "massLoss": massLoss, "peaked": peaked}
        
        return blobInfo
    
    def parse_sim_output(self, run_info={}):
        """
        Parses a BOUT.dmp.*.nc file from the output of blob2d and converts it to the EasyVVUQ
        internal dictionary based format.  The file is parsed in such a way that each column
        appears as a vector QoI in the output dictionary.

        E.g. if the file contains the LHS and `a` & `b` are specified as `output_columns` then:
        a,b
        1,2  >>>  {'a': [1, 3], 'b': [2, 4]}.
        3,4

        Parameters
        ----------
        run_info: dict
            Information about the run used to construct the absolute path to
            the blob2d output files.
        
        Returns
        -------
        outQtts (dict)
            Dictionary of quantities which may be of interest
        """
        
        out_path = os.path.join(run_info['run_dir'], self.target_filename)
        outQtts = self.get_blob_info(out_path)
        return outQtts

###############################################################################

def refine_sampling_plan(number_of_refinements, campaign, sampler, analysis, param):
    """
    Refine the sampling plan.

    Parameters
    ----------
    number_of_refinements (int)
        The number of refinement iterations that must be performed.

    Returns
    -------
    None. The new accepted indices are stored in analysis.l_norm and the admissible indices
    in sampler.admissible_idx.
    """
    
    for i in range(number_of_refinements):
        # compute the admissible indices
        sampler.look_ahead(analysis.l_norm)
        
        # run the ensemble
        campaign.execute().collate(progress_bar=True)
        
        # accept one of the multi indices of the new admissible set
        data_frame = campaign.get_collation_result()
        analysis.adapt_dimension(param, data_frame)#, method='var')

def refine_to_precision(campaign, sampler, analysis, param, tol, minrefs, maxrefs):
    """
    Refines the sampling with respect to an output variable until the error on that
    variable is below a certain tolerance.  The error is a sort of normalised 
    hierarchical surplus error but not the relative error, calculated by dividing
    the largest hierarchical surplus error of the admitted set by the initial value
    of the parameter in question.
    
    Parameters
    ----------
    campaign, sampler & analysis
        Easyvvuq objects containing their respective class info
    param
        The parameter we are refining with respect to
    tol
        The maximum desired normalised error we want after refinement, see
        dissertation for exact definition
    maxrefs
        Maximum allowed number of refinements

    Returns
    -------
    counter
        The number of refinements used
    """
    
    counter = 0
    error = 1
    while counter < minrefs or (error > tol and counter <= maxrefs):
        refine_sampling_plan(1, campaign, sampler, analysis, param)
        counter += 1
        error = analysis.get_adaptation_errors()[-1] / analysis.samples[param][0]
        print(param, " iteration ", counter)
        print(param, "error: ", error)  
    return counter

def plot_sampling(sampler, analysis):
    """
    Plots the accepted set of a dimension adaptive SC campaign over Te0-n0,
    D_vort-D_n and height-width.  This provides a resonable visualisation of the
    sampling in 6D space.
    The assumption that all six variables are included in the parameter space
    is hardcoded here.
    """
    
    fig = plt.figure(figsize=[4,12])
    ax1 = fig.add_subplot(311, xlim=[2.4, 7.6], ylim=[0.9e+18, 4.1e+18], xlabel='Te0', ylabel='n0', title='(Te0, n0) plane')
    ax2 = fig.add_subplot(312, xlim=[0, 1.01e-5], ylim=[0, 1.01e-5], xlabel='D_vort', ylabel='D_n', title='(D_vort, D_n) plane')
    ax3 = fig.add_subplot(313, xlim=[0.24, 0.76], ylim=[0.025, 0.155], xlabel='height', ylabel='width', title='(height, width) plane')
    
    accepted_grid = sampler.generate_grid(analysis.l_norm)
    ax1.plot(accepted_grid[:,0], accepted_grid[:,1], 'o')
    ax2.plot(accepted_grid[:,2], accepted_grid[:,3], 'o')
    ax3.plot(accepted_grid[:,4], accepted_grid[:,5], 'o')
    
    plt.tight_layout()
    plt.show()

def TWsurrogate(QoI, T, W, analysis):
    """
    The surrogate model for a givel QoI given fixed default values of n0,
    D_vort, D_n & height, i.e. the surrogate of a QoI projected onto a
    T-W subspace
    """
    
    return analysis.surrogate(QoI, np.array([T, 2e+18, 1e-06, 1e-06, 0.5, W]))

def plot_on_TW(QoI, analysis):
    """
    Plot a QoI (either Te0 or maxV in SI units) as it varies on the Te0 & width
    parameter subspace in 3D for  for default values of the other parameters
    """
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    T = np.arange(2.5, 7.5, 0.05)
    W = np.arange(0.03, 0.15, 0.0012)
    T, W = np.meshgrid(T, W)
    
    Z = np.zeros((100, 100))
    for t in range(len(T)):
        for w in range(len(W)):
            if QoI == "maxV":
                Z[w][t] = TWsurrogate(QoI, T[w][t], W[w][t], analysis)*np.sqrt(const.e*t/const.m_p)
            else:
                Z[w][t] = TWsurrogate(QoI, T[w][t], W[w][t], analysis)*np.sqrt(const.m_p*t/const.e)/0.35
    
    ax.plot_surface(T, W, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel("Te0 (eV)")
    ax.set_ylabel("Blob width (m)")
    if QoI == "maxV": ax.set_zlabel("Max CoM velocity (m/s)")
    else: ax.set_zlabel("Blob max displacement (m)")
    
    plt.show()

def plot_on_T(QoI, analysis):
    """
    Plot a QoI (either Te0 or maxV in SI units) as it varies on the Te0 axis for
    default values of the other parameters
    """
    
    T = np.arange(2.5, 7.5, 0.05)
    Z = np.zeros((100))
    for t in range(len(T)):
        if QoI == "maxV":
            Z[t] = TWsurrogate(QoI, T[t], 0.09, analysis)*np.sqrt(const.e*t/const.m_p)
        else:
            Z[t] = TWsurrogate(QoI, T[t], 0.09, analysis)*np.sqrt(const.m_p*t/const.e)/0.35
    
    plt.plot(T, Z)
    plt.xlabel("Te0 (eV)")
    if QoI == "maxV": plt.ylabel("Max CoM velocity (m/s)")
    else: plt.ylabel("Blob max displacement (m)")
    plt.show()

def plot_on_W(QoI, analysis):
    """
    Plot a QoI (either Te0 or maxV in SI units) as it varies on the width axis
    for default values of the other parameters
    """

    W = np.arange(0.03, 0.15, 0.0012)
    Z = np.zeros((100))
    for w in range(len(W)):
        if QoI == "maxV":
            Z[w] = TWsurrogate(QoI, 5, W[w], analysis)*np.sqrt(const.e*5/const.m_p)
        else:
            Z[w] = TWsurrogate(QoI, 5, W[w], analysis)*np.sqrt(const.m_p*5/const.e)/0.35

    plt.plot(W, Z)
    plt.xlabel("Blob width (m)")
    if QoI == "maxV": plt.ylabel("Max CoM velocity (m/s)")
    else: plt.ylabel("Blob max displacement (m)")
    plt.show()

###############################################################################

def define_params(paramFile=None):
    """
    Defines parameters to be applied to the system.

    Parameters
    ----------
    paramFile (string)
        Name of file containing system parameters, not implemented yet.

    Returns
    -------
    params (dict)
        Dictionary of parameters, their default values and their range of uncertainty.
    vary (dict)
        Dictionary of uncertain parameters and their distributions.
    output_columns (list)
        List of the quantities extracted by the decoder which we want to return.
    template (str)
        Filename of the template to be used.
    """
    
    if paramFile == None:
        params = {
                "Te0": {"type": "float", "min": 2.5, "max": 7.5, "default": 5.0},# Ambient temperature
                "n0": {"type": "float", "min": 1.0e+18, "max": 4.0e+18, "default": 2.0e+18},# Ambient density
                "D_vort": {"type": "float", "min": 0.9e-7, "max": 1.1e-5, "default": 1.0e-6},# Viscosity
                "D_n": {"type": "float", "min": 0.9e-7, "max": 1.1e-5, "default": 1.0e-6},# Diffusion
                "height": {"type": "float", "min": 0.25, "max": 0.75, "default": 0.5},# Blob amplitude
                "width": {"type": "float", "min": 0.03, "max": 0.15, "default": 0.09},# Blob width
        }
        vary = {
                "Te0": cp.Uniform(2.5, 7.5),
                "n0": cp.Uniform(1.0e+18, 4.0e+18),
                "D_vort": cp.Uniform(1.0e-7, 1.0e-5),
                "D_n": cp.Uniform(1.0e-7, 1.0e-5),
                "height": cp.Uniform(0.25, 0.75),
                "width": cp.Uniform(0.03, 0.15)
        }
        output_columns = ["maxV", "maxX", "avgTransp", "massLoss"]
        template = 'b2d.template'
        
        return params, vary, output_columns, template
    
    else:
        # Don't plan to use parameter files but will write in code to do so if needed
        #pFile = load(paramFile)
        #params = pFile[0]
        #...
        #return params, vary, output_columns, template
        pass

def setup_campaign(name, params, output_columns, template):
    """
    Builds a campaign using the parameters provided.

    Parameters
    ----------
    params (dict)
        Dictionary of parameters, their default values and their range of uncertainty.
    output_columns (list)
        List of the quantities we want the decoder to pass out.
    template (str)
        Filename of the template to be used.

    Returns
    -------
    campaign (easyvvuq campaign object)
        The campaign, build accoring to the provided parameters.
    """
    
    # Create encoder
    encoder = uq.encoders.GenericEncoder(
            template_fname=template,
            delimiter='$',
            target_filename='BOUT.inp')
    
    # Create executor - 60 timesteps in home directory with no verbosity
    # Nice level is set to run on YPI servers
    execute = ExecuteLocal(f'nice -n 11 mpirun -np 32 {os.getcwd()}/blob2d -d ./ nout=60 -q -q -q')
    
    # Create decoder
    decoder = B2dDecoder(
            target_filename="BOUT.dmp.*.nc",
            output_columns=output_columns)
    
    # Ensure run directory exists, then pack up encoder, decoder, executor and build campaign
    if os.path.exists('outfiles')==0: os.mkdir('outfiles')
    actions = Actions(CreateRunDirectory('outfiles'), Encode(encoder), execute, Decode(decoder))
    campaign = uq.Campaign(
            name=name,
            #db_location="sqlite:///" + os.getcwd() + "/campaign.db",
            work_dir='outfiles',
            params=params,
            actions=actions)
    
    return campaign

def setup_sampler(campaign, vary):
    """
    Creates and returns an easyvvuq sampler object for an adaptive dimension stochastic
    collocation campaign using the uncertain parameters from vary, then applies it to
    the campaign object
    """
    
    sampler = uq.sampling.SCSampler(
            vary=vary,
            polynomial_order=1,
            quadrature_rule="C",
            sparse=True,
            growth=True,
            midpoint_level1=True,
            dimension_adaptive=True)
    campaign.set_sampler(sampler)
    
    return sampler

def get_analysis(campaign, sampler, output_columns):
    """
    Creates, saves and returns the analysis class which will be used on the campaign
    """
    
    frame = campaign.get_collation_result()
    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
    campaign.apply_analysis(analysis)
    analysis.save_state(f"{campaign.campaign_dir}/analysis.state")
    
    print(frame)
    
    return analysis

def load_analysis(campaign, sampler):
    """
    Loads and returns the analysis class from a previous campaign
    """
    
    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=sampler.vary.get_keys())
    analysis.load_state(f"{campaign.campaign_dir}/analysis.state")
    
    return analysis

def refine_campaign(campaign, sampler, analysis, output_columns):
    """
    Refines a campaign according to hardcoded limits and toleranf and saves the
    number of refinements to a file
    """

    atRefs = refine_to_precision(campaign, sampler, analysis, 'maxV', 0.01, 5, 10)
    mlRefs = refine_to_precision(campaign, sampler, analysis, 'maxX', 0.01, 5, 10)
    campaign.apply_analysis(analysis)
    np.savetxt('refinements.txt', np.asarray([atRefs, mlRefs]))

def analyse_campaign(campaign, sampler, analysis):
    """
    Runs a set of analyses on a provided campaign, details often change by commit.
    Currently demonstrates use of some functions which might be useful when analysing a campaign
    
    Parameters
    ----------
    campaign, sampler & analysis
        Easyvvuq objects containing their respective class info

    Returns
    -------
    None - results either printed to screen, plotted or saved to a file
    """
    
    print("Analysis start")
    #frame = campaign.campaign_db.get_results("FullSim", 1, iteration=-1)
    #results = analysis.analyse(frame)
    #pprint(frame)#.to_string())
    #print(analysis.l_norm)
    
    # Show the maximum hierarchical surplus error of the admissible set after each refinement
    print(analysis.get_adaptation_errors())
    
    # Merge accepted and admissible sets
    analysis.merge_accepted_and_admissible()
    frame = campaign.get_collation_result()
    results = analysis.analyse(frame)

    # Print the dataframe collated from the campaign object (the values of inputs and outputs at each node)    
    pprint(frame)
    
    # Print a value at some point on the parameter space for a QoI using the surrogate model
    p = np.array([7.5e+00, 2.5e+18, 5.05e-06, 5.05e-06, 5.0e-01, 3.4567228e-02])
    print("Surrogate: ", analysis.surrogate("avgTransp", p))
    
    # Show all Sobol indices for massLoss
    sobolsML = analysis.get_sobol_indices('massLoss', typ='all')
    pprint(sobolsML)
    
    # Show first order Sobol indices for maxX
    sobolsMX = analysis.get_sobol_indices('maxX').values()
    pprint(sobolsMX)
    
    # Plots the results used in dissertation
    plot_on_TW("maxV", analysis)
    plot_on_TW("maxX", analysis)
    plot_on_T("maxV", analysis)
    plot_on_T("maxX", analysis)
    plot_on_W("maxV", analysis)
    plot_on_W("maxX", analysis)
    
    # Other plots
    plot_sampling(sampler, analysis)
    analysis.adaptation_table()
    
    plt.show()

###############################################################################

def main():
    campaign_name = "FullSim"
    new_campaign = False
    if new_campaign:
        # Run a new campaign (takes about a day to run on Pinch with current settings)
        params, vary, output_columns, template = define_params()
        campaign = setup_campaign(campaign_name, params, output_columns, template)
        sampler = setup_sampler(campaign, vary)
        campaign.execute().collate(progress_bar=True)
        analysis = get_analysis(campaign, sampler, output_columns)
        refine_campaign(campaign, sampler, analysis, output_columns)
        analysis.save_state(f"{campaign.campaign_dir}/analysis.state")
    else:
        # Load an old campaign and perform analysis
        campaign = uq.Campaign(name=campaign_name, db_location="sqlite:///" + "outfiles/FullSim99d7jlfm/campaign.db")
        sampler = campaign.get_active_sampler()
        campaign.set_sampler(sampler, update=True)
        analysis = load_analysis(campaign, sampler)
        
        # Refine campaign further if neccecary
        #refine_campaign(campaign, sampler, analysis, output_columns)
        #analysis.save_state(f"{campaign.campaign_dir}/analysis.state")
        
        analyse_campaign(campaign, sampler, analysis)
    
    print("Campaign run / analysed successfuly")
    print(campaign.campaign_dir)

if __name__ == "__main__":
    main()
