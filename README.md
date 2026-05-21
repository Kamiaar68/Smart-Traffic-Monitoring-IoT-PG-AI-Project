# Smart Traffic Monitoring — IoT & Big Data Project

**Livestream source:** public YouTube traffic camera — `https://youtu.be/z545k7Tcb5o`

## Detailed Installation Procedure

**1. Install Python 3.9 or newer.** Verify:
```bash
python --version
```

**2. Clone the repository:**
```bash
git clone https://github.com/Kamiaar68/Smart-Traffic-Monitoring-IoT-PG-AI-Project.git
cd Smart-Traffic-Monitoring-IoT-PG-AI-Project
```

**3. Install dependencies:**
```bash
pip install ultralytics opencv-python yt-dlp
```
This installs `ultralytics` (which pulls in PyTorch), `opencv-python`, and `yt-dlp`. The first install may take a few minutes.

**4. Run the application:**
```bash
python IoT_Project0.4.py
```
A window opens showing the live feed with bounding boxes, IDs, the counting zone, live metrics, and any active alerts. Press **`q`** to quit.

## Output — Raw Data Results

Data is appended to `traffic_data.csv` with the following schema:

| Column | Description |
|--------|-------------|
| `timestamp` | Time the row was logged |
| `cars` | Cumulative car count |
| `trucks` | Cumulative truck count |
| `buses` | Cumulative bus count |
| `motorcycles` | Cumulative motorcycle count |
| `vehicles_per_min` | Vehicle throughput |
| `traffic_level` | LOW / MEDIUM / HIGH |
| `emission_index` | Weighted environmental impact score |

### Emission weights
| Vehicle | Weight |
|---------|--------|
| Car | 1 |
| Motorcycle | 0.5 |
| Bus | 4 |
| Truck | 3 |
