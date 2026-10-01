library(gdm)
library(terra)

# get file paths
args <- commandArgs(trailingOnly = TRUE)
site_survey_filename = args[1]
env_rast_filename = args[2]
abundance = as.logical(args[3])
fits_filename = args[4]
pca_rast_filename = args[5]

# read site-survey table
site_survey = read.csv(site_survey_filename)

# read environmental raster data
env_rast = terra::rast(env_rast_filename)

# format data for analysis
site_pair = gdm::formatsitepair(bioData=site_survey,
                                bioFormat=1,
                                abundance=abundance,
                                siteColumn='site',
                                XColumn='x',
                                YColumn='y',
                                predData=env_rast)

# fit GDM
mod = gdm::gdm(site_pair, geo=F)
print(summary(mod))

# save fitted functions
fits = as.data.frame(isplineExtract(mod))
write.csv(fits, fits_filename)

# save first <=3 PCs of GDM-transformed env space
env_rast_trans <- gdm::gdm.transform(model=mod, data=env_rast)
pca_samp <- terra::prcomp(env_rast_trans, maxcell = 5e5)
n_pcs = min(3, dim(env_rast)[3])
pca_rast <- terra::predict(env_rast_trans, pca_samp, index=1:n_pcs)
pca_rast <- terra::stretch(pca_rast)
terra::writeRaster(pca_rast, pca_rast_filename, overwrite=T)
cat("\nGDM OUTPUTS SAVED TO DISK.\n")
