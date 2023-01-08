
def classify_baygaud(path_cube, path_mask=None, path_bulk_refvf=None, bool_classify_comps=True, lim_intsn=0.0, lim_peaksn=3.0, lim_vel_cnm=4.0, lim_vel_wnm=8.0, range_bulk=1.0, multiplier_cube=1./1000., print_message=True):

	# Parameters:
	# path_cube  			: path to the cube
	# path_mask  			: path to the mask reference (not necessary)
	# path_bulk_refvf		: path to bulk reference velocity field; if None, the program will not do bulk classifying
	# bool_classify_comps	: whether to classify cold/warm/hot; if false, the program will only do filtering
	# lim_intsn  			: integrated S/N limit for cut
	# lim_peaksn 			: peak S/N limit for cut
	# lim_vel_cnm			: velocity boundary for COLD/WARM component
	# lim_vel_wnm			: velocity boundary for WARM/HOT component
	# range_bulk 			: range multiplier for bulk classification (|bulk|<range_bulk*single_vdisp)
	# multiplier_cube		: to convert m/s unit of cube to km/s

	# DEPENDENCIES:
	# scipy: pip3 install scipy
	# astropy: pip3 install astropy
	# spectral_cube: pip3 install spectral_cube

	import glob
	import os
	import shutil
	import warnings

	import numpy as np
	import scipy.stats as stats
	from astropy.io import fits
	from spectral_cube import SpectralCube

	warnings.filterwarnings('ignore', category=RuntimeWarning)

	if(os.path.exists(path_cube)==False):
		print('[CLASSIFY] Cube not found. Program ends.')
		return

	if(print_message==True):
		print("===================================================")

	wdir = os.path.dirname(path_cube)    # Working directory - where cube is
	name_cube = path_cube.split("/")[-1] # Name of the given cube

	if(print_message==True):
		print(name_cube)

	if(print_message==True):
		print('[CLASSIFY] User inputs')
		print('     Integrated S/N limit = {}'.format(lim_intsn))
		print('           Peak S/N limit = {}'.format(lim_peaksn))
		if(bool_classify_comps==True):
			print('  Velocity limit for cold = {}'.format(lim_vel_cnm))
			print('  Velocity limit for warm = {}'.format(lim_vel_wnm))
		if(path_bulk_refvf!=None):
			print('    Bulk limit multiplier = {}'.format(range_bulk))

	if(path_mask!=None and os.path.exists(path_mask)==False):
		print("[CLASSIFY] Mask reference not found. Mask won't be applied")

	if(path_bulk_refvf!=None and os.path.exists(path_bulk_refvf)==False):
		print("[CLASSIFY] Bulk reference vf not found. Bulk output won't be made.")




	n_opt = len(glob.glob(wdir+"/baygaud_output.{}/output_merged/G*".format(os.path.splitext(name_cube)[0]))) # Number of maximum Gaussians of BAYGAUD

	# Read mask if given
	if(path_mask!=None):
		data_mask = fits.getdata(path_mask)
		data_mask[data_mask==np.inf] = np.nan

	# Read bulk refvf if given
	if(path_bulk_refvf!=None):
		data_bref = fits.getdata(path_bulk_refvf)
		data_bref[data_bref==np.inf] = np.nan

	path_merged = glob.glob(wdir+"/baygaud_output.{}/output_merged".format(os.path.splitext(name_cube)[0]))[0]
	path_classified = path_merged + "/classified_intsn{}_peaksn{}".format(lim_intsn, lim_peaksn)

	# Remove the output directory if already exists
	if(os.path.exists(path_classified)):
		shutil.rmtree(path_classified)

	# Make directories
	os.mkdir(path_classified)

	os.mkdir(path_classified+"/single_gfit")
	if(bool_classify_comps==True):
		os.mkdir(path_classified+"/_1chan_cnm_gfit")
		os.mkdir(path_classified+"/_cnm_wnm_gfit")
		os.mkdir(path_classified+"/_wnm_hot_gfit")
	for i in range(n_opt):
		os.mkdir(path_classified+"/G0{}g0{}".format(n_opt, i+1))
	
	if(path_bulk_refvf!=None):
		os.mkdir(path_classified+"/bulk")
		os.mkdir(path_classified+"/non_bulk")
		os.mkdir(path_classified+"/bulk/core_bulk")

	channel_width = np.abs(fits.getheader(path_cube)['CDELT3']*multiplier_cube)
	spectral_axis = SpectralCube.read(path_cube).spectral_axis.value*multiplier_cube
	
	mapnames = ['0','0.e',	# background
				'1','1.e',	# integrated intensity
				'2','2.e',	# velocity dispersion
				'3','3.e',	# centroid velocity
				'4','4.e',	
				'5','5.e',	# N_gauss
				'6','6.e',	# Noise (RMS)
				'7','7.e',	# Peak S/N
				'8','8.e',	
				'9','9.e',	
				'10','10.e']

	dict_hdr = {}   # Header info will be stored
	dict_data = {}  # Map data will be stored

	# ==============================================================================
	# FILTERING single_gfit
	# ==============================================================================

	# Read BAYGAUD outputs
	for mapname in mapnames:
		path_map = glob.glob(path_merged+"/single_gfit/*.{}.fits".format(mapname))[0]

		data_map = fits.getdata(path_map)
		data_map[data_map==np.inf] = np.nan

		dict_hdr['S'+mapname] = fits.getheader(path_map)
		dict_data['S'+mapname] = data_map

		# del path_map
		del data_map
	
	# Pre-define headers for maps those will be newly generated
	dict_hdr['Sintsnmap'] = fits.getheader(path_map)
	dict_hdr['SN_channel'] = fits.getheader(path_map)
	dict_hdr['Sintsnmap']['BUNIT'] = '[int s/n]'
	dict_hdr['SN_channel']['BUNIT'] = '[N_channel]'	

	median_rms = np.nanmedian(dict_data['S6'])  # Median value of noise of single_gfit
												# Used to calculate the integrated S/N

	# Calculate N_channel: Number of channels occupied by a component (+- 3 sigma)
	lowend  = dict_data['S3'] - 3*dict_data['S2']
	highend = dict_data['S3'] + 3*dict_data['S2']
	min_specaxis = np.nanmin(spectral_axis)
	max_specaxis = np.nanmax(spectral_axis)
	r_min = np.where(lowend < min_specaxis, min_specaxis, lowend)   # Restrict high/low end values inside the spectral axis
	r_max = np.where(highend > max_specaxis, max_specaxis, highend)

	dict_data['SN_channel'] = (r_max-r_min)/channel_width
	dict_data['SN_channel'][dict_data['SN_channel']==np.inf] = 0
	dict_data['SN_channel'][np.isnan(dict_data['SN_channel'])] = 0
	dict_data['SN_channel'] = np.float64(np.int64(dict_data['SN_channel']))
	dict_data['SN_channel'][dict_data['SN_channel']==0] = np.nan

	# Calculate integrated S/N: [INTENSITY] / ([median(RMS)] * [sqrt(N_channel)] * channel width)
	dist_norm = stats.norm(dict_data['S3'], dict_data['S2'])
	dict_data['S1'] = dict_data['S1'] * (dist_norm.cdf(r_max) - dist_norm.cdf(r_min)) * multiplier_cube # Redefine intensity with calculated high/lowends above
	dict_data['Sintsnmap'] = dict_data['S1']/(median_rms*np.sqrt(dict_data['SN_channel'])*channel_width)

	del lowend
	del highend
	del min_specaxis
	del max_specaxis
	del r_min
	del r_max
	del dist_norm

	# Filtering conditions
	cond1 = dict_data['Sintsnmap'] < lim_intsn		# Remove if intsn < intsn limit
	cond2 = dict_data['S7']        < lim_peaksn		# Remove if peaksn < peaksn limit
	cond3 = dict_data['S2']        < channel_width	# Remove if fitted vdisp is smaller than channel width
	cond4 = dict_data['S2']        > 99.0			# Remove if fitted vdisp is crazy
	cond = (cond1|cond2|cond3|cond4)				# Remove if one of above conditions is met

	# Make map containing info why a component is removed
	dict_data['Swhy'] = np.zeros((4, np.shape(cond1)[0], np.shape(cond1)[1]))
	dict_data['Swhy'][0] = np.where(cond1, dict_data['Sintsnmap'], dict_data['Swhy'][0])
	dict_data['Swhy'][1] = np.where(cond2, dict_data['S7'],     dict_data['Swhy'][1])
	dict_data['Swhy'][2] = np.where(cond3, dict_data['S2'],     dict_data['Swhy'][2])
	dict_data['Swhy'][3] = np.where(cond4, dict_data['S2'],     dict_data['Swhy'][3])

	del cond1
	del cond2
	del cond3
	del cond4
	
	# Filter maps
	for mapname in mapnames:
		if(mapname=='5'):
			continue
		dict_data['S'+mapname][cond] = np.nan

	dict_data['S5'][cond] -= 1 # Subtract 1 from Ngauss map if a component is removed
	dict_data['S1.e'] * multiplier_cube

	# Mask maps if mask reference is given
	if(path_mask!=None):
		for key in dict_data.keys():
			if(key=='Swhy'): continue
			dict_data[key][np.isnan(data_mask)] = np.nan

	del cond

	# Save filtered images
	for mapname in mapnames:
		fits.writeto(path_classified+"/single_gfit/{}.single_gfit.{}.fits".format(name_cube, mapname), dict_data['S'+mapname], dict_hdr['S'+mapname], overwrite=True)

	fits.writeto(path_classified+"/single_gfit/{}.single_gfit.cond.fits".format(name_cube), dict_data['Swhy'], overwrite=True)
	fits.writeto(path_classified+"/single_gfit/{}.single_gfit.N_channel.fits".format(name_cube), dict_data['SN_channel'], dict_hdr['SN_channel'], overwrite=True)
	fits.writeto(path_classified+"/single_gfit/{}.single_gfit.intsnmap.fits".format(name_cube), dict_data['Sintsnmap'], dict_hdr['Sintsnmap'], overwrite=True)

	if(print_message==True):
		print('[CLASSIFY] Single GFIT filtering complete')
	
	# ==============================================================================
	# FILTERING Gs
	# ==============================================================================

	# Iterate over maximum number of fitted Gaussian
	for i in range(n_opt):

		Gn = 'G{}'.format(i+1)

		# Read BAYGAUD output
		for mapname in mapnames:
			path_map = glob.glob(path_merged+"/G*g0{}/*.{}.fits".format(i+1,mapname))[0]

			data_map = fits.getdata(path_map)
			data_map[data_map==np.inf] = np.nan

			if(path_mask!=None):
				data_map[np.isnan(data_mask)] = np.nan

			dict_hdr[Gn+mapname] = fits.getheader(path_map)
			dict_data[Gn+mapname] = data_map

			del path_map
			del data_map

		lowend  = dict_data[Gn+'3'] - 3*dict_data[Gn+'2']
		highend = dict_data[Gn+'3'] + 3*dict_data[Gn+'2']
		min_specaxis = np.nanmin(spectral_axis)
		max_specaxis = np.nanmax(spectral_axis)
		r_min = np.where(lowend < min_specaxis, min_specaxis, lowend)
		r_max = np.where(highend > max_specaxis, max_specaxis, highend)

		dict_data[Gn+'N_channel'] = (r_max-r_min)/channel_width
		dict_data[Gn+'N_channel'][dict_data[Gn+'N_channel']==np.inf] = 0
		dict_data[Gn+'N_channel'][np.isnan(dict_data[Gn+'N_channel'])] = 0
		dict_data[Gn+'N_channel'] = np.float64(np.int64(dict_data[Gn+'N_channel']))
		dict_data[Gn+'N_channel'][dict_data[Gn+'N_channel']==0] = np.nan

		dist_norm = stats.norm(dict_data[Gn+'3'], dict_data[Gn+'2'])
		dict_data[Gn+'1'] = dict_data[Gn+'1'] * (dist_norm.cdf(r_max) - dist_norm.cdf(r_min))/1000.
		dict_data[Gn+'intsnmap'] = dict_data[Gn+'1']/(median_rms*np.sqrt(dict_data[Gn+'N_channel'])*channel_width)

		del lowend
		del highend
		del min_specaxis
		del max_specaxis
		del r_min
		del r_max
		del dist_norm

		cond1 = dict_data[Gn+'intsnmap'] < lim_intsn
		cond2 = dict_data[Gn+'7']     < lim_peaksn
		cond3 = dict_data[Gn+'2']     < channel_width
		cond4 = dict_data[Gn+'2']     > 99.0
		cond = (cond1|cond2|cond3|cond4)

		dict_data[Gn+'why'] = np.zeros((4, np.shape(cond1)[0], np.shape(cond1)[1]))
		dict_data[Gn+'why'][0] = np.where(cond1, dict_data[Gn+'intsnmap'], dict_data[Gn+'why'][0])
		dict_data[Gn+'why'][1] = np.where(cond2, dict_data[Gn+'7'],     dict_data[Gn+'why'][1])
		dict_data[Gn+'why'][2] = np.where(cond3, dict_data[Gn+'2'],     dict_data[Gn+'why'][2])
		dict_data[Gn+'why'][3] = np.where(cond4, dict_data[Gn+'2'],     dict_data[Gn+'why'][3])

		del cond1
		del cond2
		del cond3
		del cond4

		for mapname in mapnames:
			if(mapname=='5'):
				continue
			dict_data[Gn+mapname][cond] = np.nan

		dict_data['G15'][cond] -= 1
		dict_data[Gn+'1.e'] /= 1000.

		del cond

	# If a pixel that originally had more than component reduced to 1 due to filtering,
	# -> Use single_gfit result for the pixel
	for i in range(n_opt):
		Gn = 'G{}'.format(i+1)
				   # Ngauss == 1	   # A component's intensity>0
		filter = ((dict_data['G15']==1) & (dict_data[Gn+'1']>0) & (~np.isnan(dict_data['S1'])))

		for mapname in mapnames:
			if(mapname=='5'): continue
			dict_data[Gn+mapname] = np.where(filter, dict_data['S'+mapname], dict_data[Gn+mapname])

	# If a pixel that originally had more than component reduced to 0 due to filtering,
	# -> Use single_gfit result for the pixel
	for mapname in mapnames:
		if(mapname=='5'): continue
		dict_data['G1'+mapname] = np.where(dict_data['G15']==0, dict_data['S'+mapname], dict_data['G1'+mapname])

	filter = ((dict_data['G15']==0) & (dict_data['S1']>0) & (dict_data['S1']<10000))
	dict_data['G15'] = np.where(filter, 1, dict_data['G15'])

	# Define an empty array to calculate total intensity of fitted Gaussians in a pixel
	dict_data['Gn1_tot'] = np.zeros_like(dict_data['G11'])

	for i in range(n_opt):
		Gn = 'G{}'.format(i+1)

		# Save maps
		for mapname in mapnames:		
			fits.writeto(path_classified+"/G0{}g0{}/{}.bvf.g{}.{}.fits".format(n_opt, i+1, name_cube, i, mapname), dict_data[Gn+mapname], dict_hdr['S'+mapname], overwrite=True)

		fits.writeto(path_classified+"/G0{}g0{}/{}.bvf.g{}.cond.fits".format(n_opt, i+1, name_cube, i), dict_data[Gn+'why'], overwrite=True)
		fits.writeto(path_classified+"/G0{}g0{}/{}.bvf.g{}.N_channel.fits".format(n_opt, i+1, name_cube, i), dict_data[Gn+'N_channel'], dict_hdr['SN_channel'], overwrite=True)
		fits.writeto(path_classified+"/G0{}g0{}/{}.bvf.g{}.intsnmap.fits".format(n_opt, i+1, name_cube, i), dict_data[Gn+'intsnmap'], dict_hdr['Sintsnmap'], overwrite=True)

		# Calculate total intensity of fitted Gaussians in a pixel
		dict_data['Gn1_tot'] += np.where(np.isnan(dict_data['G{}1'.format(i+1)]), 0, dict_data['G{}1'.format(i+1)])

	# Save total intensity
	dict_data['Gn1_tot'][dict_data['Gn1_tot']==0] = np.nan
	fits.writeto(path_classified+"/G0{}g01/{}.bvf.all.1.fits".format(n_opt, name_cube), dict_data['Gn1_tot'], dict_hdr['S1'], overwrite=True)

	# Remove unnecessary variables
	for i in range(n_opt):
		for mapname in mapnames:
			del dict_hdr['G{}{}'.format(i+1,mapname)]

	del dict_data['Gn1_tot']

	if(print_message==True):
		print('[CLASSIFY] Optimal GFIT filtering complete')

	# ==============================================================================
	# CLASSIFYING COMPONETNS (COLD/WARM/HOT)
	# ==============================================================================

	if(bool_classify_comps==True):
		comps = ['_1chan_cnm', '_cnm_wnm', '_wnm_hot'] # Cold, warm, hot
		mapnames = np.append(mapnames, 'intsnmap')
		mapnames = np.append(mapnames, 'N_channel')

		# Iterate over components
		for comp in comps:

			# Define empty arrays
			dict_data['CXn1_tot'] = np.zeros_like(dict_data['S1']) # Total intensity of a component in the pixel
			dict_data['CXn2_tot'] = np.zeros_like(dict_data['S1']) # Sum of vdisps of a component in the pixel
			dict_data['CXn2_addcount'] = np.zeros_like(dict_data['S1']) # Add count of CXn2_tot

			for i in range(n_opt):

				# Define filters
				#	Classify a component as COLD if vdisp < lim_vel_cnm
				if(comp==comps[0]): filter = (dict_data['G{}2'.format(i+1)] <= lim_vel_cnm)
				#	Classity a component as WARM if lim_vel_cnm < vdisp < lim_vel_wnm
				if(comp==comps[1]): filter = (dict_data['G{}2'.format(i+1)] > lim_vel_cnm) & (dict_data['G{}2'.format(i+1)] <= lim_vel_wnm)
				#	Classify a component as HOT if vdisp > lim_vel_wnm
				if(comp==comps[2]): filter = dict_data['G{}2'.format(i+1)] > lim_vel_wnm

				# Classify components using filter above
				for mapname in mapnames:
					varname = 'C{}{}{}'.format(comp,i+1,mapname)
					dict_data[varname] = np.where(filter, dict_data['G{}{}'.format(i+1,mapname)], np.nan)

				del varname
			
				# Calculate total intensity and two for mean_vdisp
				dict_data['CXn1_tot'] += np.where(np.isnan(dict_data['C{}{}1'.format(comp,i+1)]), 0, dict_data['C{}{}1'.format(comp,i+1)])
				dict_data['CXn2_tot'] += np.where(np.isnan(dict_data['C{}{}2'.format(comp,i+1)]), 0, dict_data['C{}{}2'.format(comp,i+1)])
				dict_data['CXn2_addcount'] += np.where(np.isnan(dict_data['C{}{}2'.format(comp,i+1)]), 0, 1)

				# Save data by components
				for mapname in mapnames:
					varname = 'C{}{}{}'.format(comp,i+1,mapname)
					writename = path_classified+"/{}_gfit/{}.{}_{}.{}.fits".format(comp, name_cube, comp, i+1, mapname)
					fits.writeto(writename, dict_data[varname], dict_hdr['S{}'.format(mapname)], overwrite=True)

					del dict_data[varname]

				del varname
				del writename

			# Calculate mean vdisp in a pixel: Sum of vdisp in a pixel / add count
			dict_data['CXn2_mean'] = dict_data['CXn2_tot']/dict_data['CXn2_addcount']

			del dict_data['CXn2_tot']
			del dict_data['CXn2_addcount']

			# Save total intensity and mean vdisp
			writename = path_classified+"/{}_gfit/{}.{}_{{}}.{{}}.fits".format(comp, name_cube, comp)
			fits.writeto(writename.format('all', '1'), dict_data['CXn1_tot'], dict_hdr['S1'], overwrite=True)
			fits.writeto(writename.format('mean','2'), dict_data['CXn2_mean'], dict_hdr['S2'], overwrite=True)

			del dict_data['CXn1_tot']
			del dict_data['CXn2_mean']

		if(print_message==True):
			print('[CLASSIFY] COLD/WARM/HOT classification complete')

	# ==============================================================================
	# CLASSIFYING COMPONETNS (BULK/NON-BULK)
	# ==============================================================================

	# Skip if bulk refvf is not given
	if(path_bulk_refvf!=None):

		# Define an empty array to store deviations of bulk components from refvf
		dict_data['dev_bulk'] = np.zeros((n_opt, np.shape(dict_data['S1'])[0], np.shape(dict_data['S1'])[1]))

		# Classify components into bulk and non-bulk
		for i in range(n_opt):
			for mapname in mapnames:
				bulkname = 'bulk{}{}'.format(i+1,mapname)
				nblkname = 'nblk{}{}'.format(i+1,mapname)
				# If a component's deviation from refvf < range bulk * vdisp (single_gfit):
				# -> Classify as BULK, otherwise NON-BULK
				filter = np.abs(data_bref - dict_data['G{}3'.format(i+1)]) <= range_bulk * dict_data['S2']
				dict_data[bulkname] = np.where(filter, dict_data['G{}{}'.format(i+1,mapname)], np.nan)
				dict_data[nblkname] = np.where(~filter,dict_data['G{}{}'.format(i+1,mapname)], np.nan)

			# Calculate deviations
			dict_data['dev_bulk'][i] = np.abs(dict_data['bulk{}3'.format(i+1)] - data_bref)
		
		# Locate bulk component nearest to the refvf => CORE BULK
		dict_data['argwhere_core'] = np.argmin(dict_data['dev_bulk'], axis=0)

		for i in range(n_opt):
			for mapname in mapnames:
				dict_data['corebulk{}'.format(mapname)] = np.zeros_like(dict_data['S1'])
				for j in range(n_opt):
					dict_data['corebulk{}'.format(mapname)] += np.where(dict_data['argwhere_core']==j, dict_data['bulk{}{}'.format(j+1, mapname)], 0)

				writename = path_classified+'/{{0:}}/{}.{{0:}}{}.{}.fits'.format(name_cube, i+1, mapname)
				fits.writeto(writename.format('bulk'), dict_data['bulk{}{}'.format(i+1,mapname)], dict_hdr['S'+str(i+1)], overwrite=True)
				fits.writeto(writename.format('non_bulk'), dict_data['nblk{}{}'.format(i+1,mapname)], dict_hdr['S'+str(i+1)], overwrite=True)

		for mapname in mapnames:
			writename = path_classified+'/bulk/core_bulk/{}.bulk{}.fits'.format(name_cube, mapname)
			fits.writeto(writename, dict_data['corebulk{}'.format(mapname)], dict_hdr['S'+mapname], overwrite=True)
		
		if(print_message==True):
			print('[CLASSIFY] BULK/NON-BULK classification complete')

	if(print_message==True):
		print('[CLASSIFY] Output made at {}'.format(path_classified))
		print('[CLASSIFY] Program ends')

	# for variable in dir():
	# 	if variable[0:2]!='__':
	# 		del globals()[variable]

	return

