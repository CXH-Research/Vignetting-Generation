# Vignetting-Generation

Methods to generate vignetting in images.

Combine vignetting mask and image directly. Simple but effective.

Require pillow, pytorch and torchvision.

## Command to run the program

```python3
python gen_vig.py
```

## Note

You can alter the center, radius and intensity of vignetting mask. 

The default radius is between 

```python3
max(width, height) // 1.2, max(width, height)
```

The default strength is a uniform distribution between 0.8 and 1.2.

## Example input
![](./input.png)

## Example output
![](./output.png)