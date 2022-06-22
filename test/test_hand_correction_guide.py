import tifffile as tif
from src.stardist_inference.gen_analytics import gen_hand_correction_guide

def test_hand_correction():
    seg = tif.imread(r"C:\Users\ab50\Documents\git\posfai_cell_tracking\test\folder_Cam_Long_00257.lux\klbOut_Cam_Long_00257.lux.label.tif")
    raw = tif.imread(r"C:\Users\ab50\Documents\git\posfai_cell_tracking\test\folder_Cam_Long_00257.lux\klbOut_Cam_Long_00257.lux.tif")
    exclude_id, numcells, frame_false_negative = gen_hand_correction_guide(seg, raw)
    print("Exclude IDs: ", exclude_id)
    print("Number of Cells: ", numcells)
    print("False Negative Check: ", frame_false_negative)