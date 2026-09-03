REAL-colon dataset — official description (copied verbatim from the dataset's own
`dataset_description.md`, released with the data; source: Biffi et al., *Sci Data* 11, 539
(2024), https://doi.org/10.1038/s41597-024-03359-0)

# Description
The REAL (Real-world multi-center Endoscopy Annotated video Library) - colon dataset
is composed of 60 recordings of real-world colonoscopies. The recordings comes from 4 different
clinical studies (001 to 004), each contributing with 15 videos.
For each patient/video, several clinical variables have been collected, including endoscope_brand, bowel cleanliness score (BBPS), number of surgically removed colon lesions, etc.
Each removed lesion has been annotated with a bounding box in each video frame where it appeared by trained labelers, supervised by expert gastroenterologists. Polyp information including histology, size and anatomical site has been recorded.

Key stats:
- 60 recordings, 15 for each of the 4 centers
- 2757723 total frames
- 132 removed colorectal polyps
- 351264 bounding box annotations

# Data Format
The dataset is composed by the following files:
- 60 compressed folders named `{SSS}-{VVV}_frames` with the frames from each recording
- 60 compressed folders named `{SSS}-{VVV}_annotation` with the annotations from each recordings
- video_info.csv file, a file with the metadata for each video
- lesion_info.csv, a file with the metadata for each lesion
- dataset_description.md, a readme file with information about the dataset

## Annotation file
Each xml file has this format (VOC-style):
    <annotation>
        <version_fmt>1.0<version_fmt>
        <folder>string</folder>
        <filename>SSS-VVV_t.jpg</filename>
        <source>
            <database>cosmoimd</database>
            <release>v1.0_20230228</release>
        </source>
        <size>
            <width>1240</width>
            <height>1080</height>
            <depth>3</depth>
        </size>
        <object>
            <name>lesion</name>
            <unique_id>videoname_lesionid</unique_id>
            <box_id>1</box_id>
            <bndbox>
                <xmin>540</xmin>
                <xmax>1196</xmax>
                <ymin>852</ymin>
                <ymax>1070</ymax>
            </bndbox>
        </object>
    </annotation>

## CSV files
`video_info.csv`:
- unique_video_name: name of the video in the format SSS-VVV (site-video)
- age, sex
- endoscope_brand: Fuji or Olympus
- fps
- num_frames
- num_lesions
- bbps: Boston Bowel Preparation Score

`lesion_info.csv`:
- unique_object_id: lesion id in the format unique_video_name_x
- unique_video_name
- size [mm]
- site: anatomical site where the removed lesion was found
- histology_extended
- histology_class: one of AD (adenoma), HP (hyperplastic), SSL (sessile serrated lesion),
  TSA (traditional serrated adenoma), OTHER

# Version
v1.0, 2023/02/28

# License
CC BY-NC-SA 4.0

---

## 本專案（SSL_shortcut）的補充註記

- `unique_video_name`（如 `001-001`）中的前三碼是 **cohort**（study），不是嚴格意義的
  clinical center：cohort 001 實際上是美國三家中心的池化資料。詳見研究計畫 Notion 頁 §3.1。
- 官方切分：每個 cohort 的 `001`–`010` 為 train、`011`–`012` 為 val、`013`–`015` 為 test
  （依 `VVV` 數字），見研究計畫 §3.1。
