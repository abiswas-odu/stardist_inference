"""CLI for stardist_inference."""
import click
from . import io_utils
from . import stardist_functions
import os

__version__ = "0.1"

@click.command()
@click.option(
    "--image_path", "-i",
    type=click.Path(exists=True, dir_okay=True,readable=True),
    default=False,
    help="The path to the original raw intensity image(s) in klb/h5/tif/npy format with the same extensions respectively."
         "The path can be a directory with the files in subdirectories.",
)
@click.option(
    "--output_dir","-o",
    type=click.Path(exists=True, dir_okay=True, readable=True),
    default=False,
    help="The output directory path.",
)
@click.option(
    "--model_dir","-m",
    type=click.Path(exists=True, dir_okay=True, readable=True),
    default="False",
    help="The directory containing the trained Stardist 3D model.",
)
@click.option(
    "--prob_thresh","-p", required=True, default=0.5, type=click.FLOAT,
    help="The probability threshold to be used to initialize the Stardist 3D model.",
)
@click.option(
    "--nms_thresh","-n", required=True, default=0.3, type=click.FLOAT,
    help="The nms threshold to be used to initialize the Stardist 3D model.",
)
@click.version_option(version=__version__)
def main(
        image_path: str,
        output_dir: str,
        model_dir: str,
        prob_thresh,
        nms_thresh
) -> None:
    """Main entry point for stardist_inference."""

    # Load model
    model = stardist_functions.initialize_model(model_dir,prob_thresh,nms_thresh)

    if os.path.isdir(image_path):
        result = [os.path.join(dp, f)
                  for dp, dn, filenames in os.walk(image_path)
                  for f in filenames if (os.path.splitext(f)[1] == '.klb' or
                                         os.path.splitext(f)[1] == '.h5' or
                                         os.path.splitext(f)[1] == '.tif' or
                                         os.path.splitext(f)[1] == '.npy')]
        for image_file in result:
            print("Processing image:", image_file)
            Xi = io_utils.read_image(image_file)
            axis_norm = (0, 1, 2)  # normalize channels independently
            label,detail = stardist_functions.run_3D_stardist(model, Xi, axis_norm, False, prob_thresh, nms_thresh)

            out_image_name = os.path.splitext(os.path.basename(image_file))[0] + ".label.tif"
            out_image_path = os.path.join(output_dir,out_image_name)
            io_utils.write_image_tif(label,out_image_path)
    else:
        print("Processing image:", image_path)
        Xi = io_utils.read_image(image_path)
        axis_norm = (0, 1, 2)  # normalize channels independently
        label,detail = stardist_functions.run_3D_stardist(model, Xi, axis_norm, False, prob_thresh, nms_thresh)

        out_image_name = os.path.splitext(os.path.basename(image_path))[0] + ".label.tif"
        out_image_path = os.path.join(output_dir,out_image_name)
        io_utils.write_image_tif(label,out_image_path)