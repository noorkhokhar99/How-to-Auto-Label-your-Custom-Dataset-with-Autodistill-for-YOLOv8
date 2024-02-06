import os
import supervision as sv
import cv2
import shutil
from autodistill.detection import CaptionOntology




ontology = CaptionOntology({"glasses": "Noor's glasses"})



IMAGES = "images"
DATASET_DIR_PATH = "dataset"
# delete folder if it already exists
if os.path.exists(DATASET_DIR_PATH):
    shutil.rmtree(DATASET_DIR_PATH)


from autodistill_grounded_sam import GroundedSAM

base_model = GroundedSAM(ontology=ontology)
dataset = base_model.label(
    input_folder=IMAGES, 
    extension=".jpg", 
    output_folder=DATASET_DIR_PATH)


