---
title: 'Mirror Controller Shapes Documentation'
date: 2023-10-25
draft: false
tags: 
  - maya
  - python
  - scripting
  - tool
  - documentation
---

Are you tired of manually repositioning asymmetrical components to match control curves on the opposite side in Maya? Look no further! Mirror Controller Shape is here to simplify your workflow and save you time. With just a few clicks, you can effortlessly mirror control shapes across different axes, ensuring symmetry and consistency in your projects.

{{< figure src="/mirrorControllerShapes/Banner.jpg" alt="image" caption="Mirror Controller Shapes" class="medium" >}}

## How to Get Started:

#### Step 1: Installation:
- Begin by copying the "mirrorControlShape.py" file into your Maya scripts directory. If you're a Windows user, this directory is commonly located at "C:/Users/<YourUsername>/Documents/maya/scripts/."

#### Step 2: Display the User Interface (UI):
- Easily access the user-friendly interface by importing the MirrorControlShape module. Simply paste the following code snippet into your Maya Python tab

```python

      from mirrorControlShape import MirrorControlShape 
      MirrorControlShape.show_dialog()
```

## How to Use:

{{< figure src="/mirrorControllerShapes/Mirror_process.gif" alt="image" caption="Mirror Controller Shapes" class="medium" >}}

- Select the NURBS curve that requires mirroring. You have the flexibility to choose your preferred mirroring axis. By default, it's set to XY, but you also have options for YZ and XZ mirroring.

- Define your specific search criteria and Mirror Controller Shapes will take care of the rest. For example, if you need to mirror control curves with a "Right_" prefix to match "Left_" prefixed control curves, just input your preferences. The tool seamlessly mirrors the controllers for you

{{< figure src="/mirrorControllerShapes/Ui.png" alt="image" caption="Mirror Controller Shapes UI" class="medium" >}}

Experience the convenience of perfect symmetrical controllers in your rigs.
