# Before you run the Python code snippet below, run the following command:
# pip install roboflow autodistill autodistill_grounding_dino pip install scikit-learn

from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill.helpers import sync_with_roboflow

BOX_THRESHOLD = 0.5
CAPTION_ONTOLOGY = {
    "recycling": "recycling"
}
TEXT_THRESHOLD = 0.70

model = GroundingDINO(
    ontology=CaptionOntology(CAPTION_ONTOLOGY),
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

sync_with_roboflow(
    workspace_id="hEK60iyCH1MD3HeJiB9fKrHr4Tw1",
    workspace_url="ecobn",
    project_id = "trash-sorter-luztm",
    batch_id = "sLmoKi4kvrJCgQtMSsXo",
    model = model
)
