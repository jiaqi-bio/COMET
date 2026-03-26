/**
 * Import COMET masks and class labels into QuPath.
 *
 * This script is a configurable variant of the COMET QuPath import step.
 * It reconstructs per-FOV mask detections on the whole-slide image and can
 * assign a QuPath PathClass from the `Class` column, import numeric NIMBUS or
 * COMET-derived columns as QuPath measurements, or do both in one pass.
 *
 * Tested with QuPath v0.5.x and intended to remain compatible with QuPath v0.6.0.
 *
 * The script will:
 * 1. Read field-of-view coordinates from `fov_coordinates.csv`
 * 2. Read a per-cell class table containing at least `fov`, `label`, and `Class`
 * 3. Import one label mask per FOV at the corresponding whole-slide location
 * 4. Optionally assign the QuPath classification for each imported detection
 * 5. Optionally import numeric NIMBUS or COMET-derived columns as QuPath measurements
 *
 * Expected inputs:
 * - A whole-slide image open in QuPath
 * - One mask file per FOV in the configured mask directory
 * - Matching FOV identifiers across the coordinate CSV, class CSV, and mask filenames
 * - Matching label identifiers between each mask and the `label` column in the class CSV
 *
 * Update the paths in the configuration section before running the script.
 *
 * @author Jiaqi Yang
 */

// Required imports
import qupath.lib.analysis.images.ContourTracing
import qupath.lib.io.TMAScoreImporter
import qupath.lib.objects.classes.PathClass
import qupath.lib.regions.RegionRequest
import java.io.File

// Configuration: update these paths for your own COMET output directory before running the script

// Example COMET slide directory
def slideDir = "/path/to/my_experiment/Slide1"

// Full path to the FOV coordinate table (`fov_coordinates.csv`)
def csvPathCoords = "${slideDir}/fov_coordinates.csv"

// Full path to the per-cell class table (must include `fov`, `label`, and `Class`)
def csvPathClasses = "${slideDir}/nimbus_output/nimbus_cell_table_classified_qupath.csv"

// Directory containing the exported whole-cell mask files
def maskDir = "${slideDir}/segmentation/deepcell_output"

// File suffix appended to each FOV name when locating the corresponding mask
def maskSuffix = "_whole_cell.tiff"

// Whether to assign QuPath PathClass from the `Class` column
def importClass = true

// Whether to import numeric columns from the CSV as QuPath measurements
def importMeasurements = false


// Load the class table first so classes and optional measurements can be assigned immediately after mask import
print "[1/4] Loading class table"

def allFovData = [:] // Main lookup table: Map<FOV_Name, Map<Label, RowMap>>
def csvDataClasses
try {
    csvDataClasses = TMAScoreImporter.readCSV(new File(csvPathClasses))
} catch (Exception e) {
    print "[ERROR] Failed to read the class CSV: " + e.message
    return
}

// Extract the key identifier columns from the class table
def fovList = csvDataClasses.get('fov')
def labelList = csvDataClasses.get('label')
if (fovList == null || labelList == null) {
    print "[ERROR] The class CSV must contain 'fov' and 'label' columns."
    return
}

// Treat all remaining columns as per-cell metadata, including the class column
def otherColumns = csvDataClasses.keySet().findAll { it != 'fov' && it != 'label' }
def measurementColumns = otherColumns.findAll { it != 'Class' && it != 'class' }
print "[INFO] Found ${otherColumns.size()} non-key columns: ${otherColumns}"
print "[INFO] importClass=${importClass}, importMeasurements=${importMeasurements}"
if (importMeasurements) {
    print "[INFO] Numeric measurement columns will be imported when possible."
}

// Build a nested lookup table for fast access during mask import
for (int i = 0; i < fovList.size(); i++) {
    def fov = fovList.get(i).toString()
    def label = labelList.get(i).toString()

    def rowMap = [:]
    otherColumns.each { colName ->
        def value = csvDataClasses.get(colName).get(i)
        rowMap.put(colName, value)
    }

    if (!allFovData.containsKey(fov)) {
        allFovData[fov] = [:]
    }
    allFovData[fov][label] = rowMap
}
print "[1/4] Loaded class data for ${allFovData.size()} FOVs."


// Load the coordinate table used to place each FOV mask on the whole-slide image
print "[2/4] Loading FOV coordinates"
def csvDataCoords
try {
    csvDataCoords = TMAScoreImporter.readCSV(new File(csvPathCoords))
} catch (Exception e) {
    print "[ERROR] Failed to read the coordinate CSV: " + e.message
    return
}

// Extract the required coordinate columns
def fovNameList = csvDataCoords.get('FOV_Name')
def xList = csvDataCoords.get('X')
def yList = csvDataCoords.get('Y')
def wList = csvDataCoords.get('Width')
def hList = csvDataCoords.get('Height')

if (fovNameList == null || xList == null || yList == null || wList == null || hList == null) {
    print "[ERROR] The coordinate CSV must contain 'FOV_Name', 'X', 'Y', 'Width', and 'Height' columns."
    return
}
print "[2/4] Loaded coordinates for ${fovNameList.size()} FOVs."


// Import each mask, convert labels to detections, and assign classes and optional measurements
print "[3/4] Importing masks and assigning classes/measurements"
def imageData = getCurrentImageData()
if (imageData == null) {
    print "[ERROR] Please open a whole-slide image in QuPath before running this script."
    return
}
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()
def serverPath = server.getPath()

int importedFovCount = 0
int notFoundFovCount = 0
int totalObjectsImported = 0
int totalClassesSet = 0
int totalMeasurementsMerged = 0

// Iterate over the coordinate table, since this defines where each FOV belongs on the slide
for (int i = 0; i < fovNameList.size(); i++) {

    // Read the placement information for the current FOV
    def fovName = fovNameList.get(i).toString()
    def x, y, w, h
    try {
        x = xList.get(i).toString().toInteger()
        y = yList.get(i).toString().toInteger()
        w = wList.get(i).toString().toInteger()
        h = hList.get(i).toString().toInteger()
    } catch (NumberFormatException e) {
        print "[WARN] Skipping ${fovName} because one or more coordinates are not valid integers."
        continue
    }

    // Resolve the expected mask filename for this FOV
    File maskFile = new File(maskDir, fovName + maskSuffix)

    if (maskFile.exists()) {
        print "[INFO] Processing ${fovName}"
        try {
            // Import the label mask and map it into whole-slide coordinates
            def region = RegionRequest.createInstance(serverPath, 1.0, x, y, w, h)
            def detections = ContourTracing.labelsToDetections(maskFile.toPath(), region)

            // Get the class rows associated with this FOV
            def measurementMap = allFovData.get(fovName)
            int classSetCount = 0
            int measurementSetCount = 0

            if (measurementMap != null) {
                // Match each imported detection by label and assign class and optional measurements
                detections.each { detection ->
                    def detLabel = detection.getName()
                    def rowMap = measurementMap.get(detLabel)

                    if (rowMap != null) {
                        if (importClass) {
                            // Support both `Class` and `class` column names.
                            def classString = rowMap.get('Class')
                            if (classString == null) {
                                classString = rowMap.get('class')
                            }

                            if (classString != null && !classString.toString().trim().isEmpty()) {
                                def newClass = PathClass.fromString(classString.toString())
                                detection.setPathClass(newClass)
                                classSetCount++
                            }
                        }

                        if (importMeasurements) {
                            def measurements = detection.getMeasurementList()
                            measurementColumns.each { colName ->
                                try {
                                    def value = rowMap.get(colName)
                                    if (value != null && !value.toString().trim().isEmpty()) {
                                        def numValue = Double.parseDouble(value.toString())
                                        measurements.put(colName, numValue)
                                    }
                                } catch (NumberFormatException e) {
                                    // Ignore values that cannot be converted to numeric QuPath measurements.
                                }
                            }
                            measurementSetCount++
                        }
                    }
                }
            } else {
                print "[WARN] No class rows were found for ${fovName} in the class CSV."
            }

            // Add the imported detections to the current image hierarchy
            hierarchy.addObjects(detections)

            print "[INFO] Imported ${detections.size()} detections, assigned classes for ${classSetCount} labels, and merged measurements for ${measurementSetCount} labels in ${fovName}."
            importedFovCount++
            totalObjectsImported += detections.size()
            totalClassesSet += classSetCount
            totalMeasurementsMerged += measurementSetCount

        } catch (Exception e) {
            print "[ERROR] Failed while importing ${fovName}: " + e.message
            e.printStackTrace()
        }

    } else {
        print "[WARN] Mask file not found for ${fovName}."
        print "[INFO] Expected path: " + maskFile.getAbsolutePath()
        notFoundFovCount++
    }
}

// Refresh QuPath and report a short summary
fireHierarchyUpdate()
print "[4/4] Import complete."
print "[SUMMARY] Imported ${importedFovCount} FOVs."
print "[SUMMARY] Created ${totalObjectsImported} cell detections."
print "[SUMMARY] Assigned classes for ${totalClassesSet} labels."
print "[SUMMARY] Merged measurements for ${totalMeasurementsMerged} labels."
if (notFoundFovCount > 0) {
    print "[SUMMARY] ${notFoundFovCount} mask files were not found. Check the configured paths and mask suffix."
}
