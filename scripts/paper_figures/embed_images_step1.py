"""
Step 1 of swapping the restyled figures into CHF_Research_Paper.docx:
replaces each embedded media/imageN.png with the freshly regenerated PNG,
and updates the drawing's wp:extent / a:ext so the displayed aspect ratio
matches the new image (same displayed width, cy recomputed) so nothing
looks stretched. Writes to CHF_Research_Paper_step1.docx.
"""
import re
import shutil
import zipfile
from PIL import Image

SRC = "CHF_Research_Paper.docx"
DST = "CHF_Research_Paper_step1.docx"

MAPPING = {
    "image1.png": "docs/manuscript/chf_ml_trends_chart.png",
    "image2.png": "docs/manuscript/fig_dataset_composition.png",
    "image3.png": "docs/manuscript/fig_axial_quality_profiles_nonuniform.png",
    "image4.png": "docs/manuscript/fig_pipeline_overview.png",
    "image5.png": "docs/manuscript/fig_eda_distribution_geometry.png",
    "image6.png": "docs/manuscript/fig_eda_pressure_vs_chf.png",
    "image7.png": "docs/manuscript/fig_correlation_heatmap_nrc.png",
    "image8.png": "docs/manuscript/fig_model_comparison_strategies.png",
    "image9.png": "docs/manuscript/fig_unseen_dataset_performance.png",
    "image10.png": "docs/manuscript/fig_best_by_scenario.png",
    "image11.png": "docs/manuscript/fig_pipeline_hierarchy.png",
    "image12.png": "docs/manuscript/fig_validation_strategies.png",
    "image13.png": "docs/manuscript/fig_accuracy_vs_cost.png",
    "image14.png": "docs/manuscript/fig_mlp_transformer_within10.png",
    "image15.png": "docs/manuscript/fig_parity_plot_mlp.png",
}

z_in = zipfile.ZipFile(SRC)
rels = z_in.read("word/_rels/document.xml.rels").decode("utf-8")
xml = z_in.read("word/document.xml").decode("utf-8")

relmap = dict(re.findall(r'Id="(rId\d+)" Type="[^"]*" Target="media/(image\d+\.png)"', rels))
rid_for_img = {v: k for k, v in relmap.items()}

drawing_re = re.compile(r"<w:drawing>.*?</w:drawing>", re.S)

for img_name, new_path in MAPPING.items():
    rid = rid_for_img[img_name]
    new_im = Image.open(new_path)
    new_ratio = new_im.width / new_im.height

    target_m = None
    for m in drawing_re.finditer(xml):
        block = m.group(0)
        if f'r:embed="{rid}"' in block:
            target_m = m
            break
    assert target_m, f"drawing block not found for {img_name} ({rid})"
    block = target_m.group(0)

    cx_m = re.search(r'<wp:extent cx="(\d+)" cy="(\d+)"/>', block)
    cx = int(cx_m.group(1))
    new_cy = round(cx / new_ratio)

    new_block = re.sub(r'(<wp:extent cx="\d+" cy=")\d+("/>)', rf'\g<1>{new_cy}\g<2>', block, count=1)
    new_block = re.sub(r'(<a:ext cx="\d+" cy=")\d+("/>)', rf'\g<1>{new_cy}\g<2>', new_block, count=1)
    xml = xml[:target_m.start()] + new_block + xml[target_m.end():]
    print(f"{img_name}: cx={cx} new_cy={new_cy} (ratio {new_ratio:.3f})")

z_in.close()

# rewrite zip: same files, swapped document.xml and media bytes
z_in = zipfile.ZipFile(SRC)
with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z_out:
    for item in z_in.infolist():
        if item.filename == "word/document.xml":
            z_out.writestr(item, xml)
        elif item.filename.startswith("word/media/") and item.filename.split("/")[-1] in MAPPING:
            img_name = item.filename.split("/")[-1]
            with open(MAPPING[img_name], "rb") as f:
                z_out.writestr(item, f.read())
        else:
            z_out.writestr(item, z_in.read(item.filename))
z_in.close()
print(f"wrote {DST}")
