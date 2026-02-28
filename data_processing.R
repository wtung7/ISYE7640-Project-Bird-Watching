# ============================================================
# ISyE 7406 - Species Distribution Modeling
# eBird Data Processing Script
# ============================================================
install.packages(c("auk", "terra", "sf", "tigris", "spThin", "dplyr"))

library(auk)
library(terra)
library(sf)
library(tigris)
library(spThin)
library(dplyr)

# ---- Set your base project path ----------------------------
base_path <- "/Users/agiri/Downloads/7406 proj"

# ============================================================
# STEP 1: STATE BOUNDARIES
# ============================================================
states_all <- states(cb = TRUE)
pnw        <- states_all[states_all$NAME %in% c("Washington", "Oregon", "Idaho"), ]
pnw_vect   <- vect(pnw)
pnw_sf     <- st_union(st_as_sf(pnw))

# ============================================================
# STEP 2: WORLDCLIM
# ============================================================
worldclim <- rast(list.files(file.path(base_path, "wc2.1_2.5m_bio"),
                             pattern = "\\.tif$",
                             full.names = TRUE))

pnw_vect       <- project(pnw_vect, crs(worldclim))
worldclim_crop <- crop(worldclim, pnw_vect)
worldclim_pnw  <- mask(worldclim_crop, pnw_vect)

# verify
plot(worldclim_pnw[[1]], main = "WorldClim BIO1 - Mean Annual Temp (PNW)")
cat("WorldClim loaded successfully --", nlyr(worldclim_pnw), "layers\n")

# ============================================================
# STEP 3: eBIRD PROCESSING
# ============================================================

# species code lookup -- matches folder name to eBird species code
species_info <- data.frame(
  folder_name  = c("american robin", "stellers jay", "red tailed hawk",
                   "spotted towhee", "western meadowlark"),
  species_code = c("amerob", "stejay", "rethaw", "spotow", "wesmea"),
  common_name  = c("American Robin", "Steller's Jay", "Red-tailed Hawk",
                   "Spotted Towhee", "Western Meadowlark"),
  stringsAsFactors = FALSE
)

states_codes <- c("OR", "WA", "ID")

# -- Function: build EBD and sampling file paths for one species + state --
get_ebd_path <- function(base, folder, code, state) {
  subfolder     <- paste0("ebd_US-", state, "_", code, "_202001_202212_smp_relJan-2026")
  ebd_file      <- file.path(base, folder, subfolder, paste0(subfolder, ".txt"))
  sampling_file <- file.path(base, folder, subfolder, paste0(subfolder, "_sampling.txt"))
  list(ebd = ebd_file, sampling = sampling_file)
}

# -- Function: process one species across all three states --
process_species <- function(folder_name, species_code, common_name) {
  
  cat("\n---- Processing:", common_name, "----\n")
  
  # build file paths for all three states
  paths <- lapply(states_codes, function(s) {
    get_ebd_path(base_path, folder_name, species_code, s)
  })
  
  # verify files exist
  for (p in paths) {
    if (!file.exists(p$ebd))      warning("Missing EBD file: ", p$ebd)
    if (!file.exists(p$sampling)) warning("Missing sampling file: ", p$sampling)
  }
  
  # read, zero-fill, cap, and filter each state individually before combining
  # capping per state prevents memory overflow when bind_rows combines them
  all_states <- lapply(paths, function(p) {
    
    if (!file.exists(p$ebd) || !file.exists(p$sampling)) return(NULL)
    cat("  Reading:", basename(p$ebd), "\n")
    
    out_ebd      <- tempfile()
    out_sampling <- tempfile()
    
    auk_ebd(p$ebd, file_sampling = p$sampling) |>
      auk_protocol(c("Stationary", "Traveling")) |>
      auk_bbox(bbox = c(-124.848974, 41.988057, -111.043564, 49.002494)) |>
      auk_complete() |>
      auk_filter(file          = out_ebd,
                 file_sampling = out_sampling,
                 overwrite     = TRUE)
    
    zerofilled          <- auk_zerofill(out_ebd, out_sampling, collapse = TRUE)
    zerofilled$presence <- as.integer(zerofilled$species_observed)
    
    # cap each state to 7,000 presences and 7,000 absences before combining
    # this prevents the full 3-state dataset from ever loading into memory at once
    pres_idx <- which(zerofilled$presence == 1)
    abs_idx  <- which(zerofilled$presence == 0)
    
    if (length(pres_idx) > 2000) pres_idx <- sample(pres_idx, 2000)
    if (length(abs_idx)  > 2000) abs_idx  <- sample(abs_idx,  2000)
    
    zerofilled[c(pres_idx, abs_idx), ]
  })
  
  # drop any NULL states (missing files) and combine
  all_states   <- Filter(Negate(is.null), all_states)
  ebd_combined <- bind_rows(all_states)
  cat("  Total records across 3 states:", nrow(ebd_combined), "\n")
  
  # cap presence records at 20,000 so all species are comparable
  n_presence <- sum(ebd_combined$presence == 1)
  n_absence  <- sum(ebd_combined$presence == 0)
  cat("  Presences:", n_presence, "| Absences:", n_absence, "\n")
  
  presence_idx <- which(ebd_combined$presence == 1)
  absence_idx  <- which(ebd_combined$presence == 0)
  
  if (n_presence > 6000) {
    presence_idx <- sample(presence_idx, 6000)
  }
  
  n_keep <- length(presence_idx)
  if (n_absence > n_keep) {
    absence_idx <- sample(absence_idx, n_keep)
  }
  
  ebd_combined <- ebd_combined[c(presence_idx, absence_idx), ]
  cat("  After capping -- presences:", sum(ebd_combined$presence == 1), "| absences:", sum(ebd_combined$presence == 0), "\n")
  
  
  # keep only necessary columns
  ebd_clean <- ebd_combined |>
    select(presence, longitude, latitude,
           observation_date, protocol_name,
           duration_minutes, effort_distance_km,
           number_observers, all_species_reported)
  
  # clip to exact state boundaries (removes any points outside WA/OR/ID)
  ebd_sf  <- st_as_sf(ebd_clean, coords = c("longitude", "latitude"), crs = 4326)
  pnw_sf  <- st_transform(pnw_sf, st_crs(ebd_sf))
  ebd_pnw <- ebd_sf[st_within(ebd_sf, pnw_sf, sparse = FALSE), ]
  cat("  Records after clipping to PNW:", nrow(ebd_pnw), "\n")
  
  # spatial thinning on presence points only to reduce sampling bias
  presence_pts <- ebd_pnw[ebd_pnw$presence == 1, ]
  absence_pts  <- ebd_pnw[ebd_pnw$presence == 0, ]
  
  coords <- st_coordinates(presence_pts)
  
  thinned <- thin(
    loc.data  = data.frame(X = coords[, 1], Y = coords[, 2], species = "target"),
    lat.col   = "Y",
    long.col  = "X",
    spec.col  = "species",
    thin.par  = 2.5,    # ~1 WorldClim grid cell at 2.5 arc-minutes
    reps      = 1,
    write.files = FALSE,
    locs.thinned.list.return = TRUE
  )[[1]]
  
  cat("  Presence records after thinning:", nrow(thinned), "\n")
  cat("  Absence records:", nrow(absence_pts), "\n")
  
  # rebuild thinned presence points as sf
  presence_thinned <- st_as_sf(thinned, coords = c("Longitude", "Latitude"), crs = 4326)
  presence_thinned$presence <- 1
  
  # combine thinned presences with absences
  ebd_final <- bind_rows(presence_thinned, absence_pts)
  
  return(ebd_final)
}

# -- Process all five species --
robin      <- process_species("american robin",    "amerob", "American Robin")
jay        <- process_species("stellers jay",      "stejay",  "Steller's Jay")
hawk       <- process_species("red tailed hawk",   "rethaw",  "Red-tailed Hawk")
towhee     <- process_species("spotted towhee",    "spotow",  "Spotted Towhee")
meadowlark <- process_species("western meadowlark","wesmea",  "Western Meadowlark")

# ============================================================
# STEP 4: EXTRACT ENVIRONMENTAL VALUES AT OCCURRENCE POINTS
# ============================================================
# Pull WorldClim values at each presence/absence point
# This creates the feature matrix

extract_env <- function(species_sf, common_name) {
  
  cat("Extracting environmental values for:", common_name, "\n")
  
  # convert to terra format
  pts <- vect(species_sf)
  
  # extract WorldClim values at each point
  wc_vals <- extract(worldclim_pnw, pts, ID = FALSE)
  
  # combine with presence/absence column
  coords     <- st_coordinates(species_sf)
  env_data   <- cbind(
    data.frame(
      presence  = species_sf$presence,
      longitude = coords[, 1],
      latitude  = coords[, 2]
    ),
    wc_vals
  )
  
  env_data$species <- common_name
  
  # remove rows with any NA environmental values
  env_data <- na.omit(env_data)
  cat("  Final records:", nrow(env_data),
      "(", sum(env_data$presence), "presences,",
      sum(env_data$presence == 0), "absences )\n")
  
  return(env_data)
}

robin_env      <- extract_env(robin,      "American Robin")
jay_env        <- extract_env(jay,        "Steller's Jay")
hawk_env       <- extract_env(hawk,       "Red-tailed Hawk")
towhee_env     <- extract_env(towhee,     "Spotted Towhee")
meadowlark_env <- extract_env(meadowlark, "Western Meadowlark")

# ============================================================
# STEP 5: SAVE PROCESSED DATA
# ============================================================
output_path <- file.path(base_path, "processed")
dir.create(output_path, showWarnings = FALSE)

write.csv(robin_env,      file.path(output_path, "robin_env.csv"),      row.names = FALSE)
write.csv(jay_env,        file.path(output_path, "jay_env.csv"),        row.names = FALSE)
write.csv(hawk_env,       file.path(output_path, "hawk_env.csv"),       row.names = FALSE)
write.csv(towhee_env,     file.path(output_path, "towhee_env.csv"),     row.names = FALSE)
write.csv(meadowlark_env, file.path(output_path, "meadowlark_env.csv"), row.names = FALSE)