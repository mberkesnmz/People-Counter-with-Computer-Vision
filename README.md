# People Counter with Computer Vision

Automatically counts people inside a defined zone using real-time video analysis.

![demo](demo.gif)
---

## What It Does

The system tracks people entering and leaving that specific region, showing the current count on screen in real time.

Instead of using a single point, the system samples the person's entire foot line — making detection more accurate when someone is partially inside the zone.

Can be used anywhere you need to monitor how many people are in a specific area — crowded venues, narrow passages, waiting areas, and more.

---

## How It Looks

- 🟢 Green box → person inside the zone
- 🟡 Yellow line → foot line used to determine if a person is inside the zone
- 🔴 Red box → person outside the zone  
- Live people count shown in the top left corner

---

## Setup

```bash
pip install ultralytics opencv-python numpy
```

Set your video path and zone coordinates in `main.py`, then run.

Press `ESC` to exit.
