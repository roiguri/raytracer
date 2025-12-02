# **Ray Tracer Implementation Plan**

This document outlines a modular, step-by-step engineering roadmap for building the Ray Tracer. The plan is divided into 6 distinct phases, moving from basic infrastructure to complex rendering features.

## **Phase 1: Infrastructure & Ray Generation**

**Goal:** Establish the coordinate system and transform 2D pixels into 3D rays.

### **Directory Structure Updates**

* camera.py:  
  * Update \_\_init\_\_ to pre-calculate camera basis vectors (forward, right, up).  
  * Implement method get\_ray(pixel\_x, pixel\_y) which returns a Ray Origin (P0) and Normalized Direction (V).  
* ray\_tracer.py:  
  * Add vector math helper functions (dot product, normalize, etc.) if not using raw NumPy heavily yet.  
  * Initialize the Camera properly with scene settings.

### **Tasks**

1. **Camera Basis:** Calculate the camera's forward vector (look\_at \- position), and use cross products with the up\_vector to find the true right and up vectors orthogonal to the viewing direction.  
2. **Screen Mapping:** Map pixel coordinates $(x, y)$ to the physical dimensions of the image plane located at screen\_distance.  
3. **Ray Casting:** Construct rays starting at the camera position and passing through the center of each pixel on the screen plane.

### **Measurable Testing**

1. **Center Ray Test:** Print the direction vector for the center pixel $(Width/2, Height/2)$. It must align exactly with normalize(look\_at \- position).  
2. **Corner Divergence:** Print ray directions for pixels $(0,0)$ and $(Width, Height)$. Ensure they diverge symmetrically from the center ray.

## **Phase 2: Geometric Intersections**

**Goal:** Implement mathematical formulas to detect ray-object collisions.

### **Directory Structure Updates**

* surfaces/sphere.py: Implement intersect(ray) $\\rightarrow$ returns (t, point, normal).  
* surfaces/infinite\_plane.py: Implement intersect(ray).  
* surfaces/cube.py: Implement intersect(ray) using the **Slabs Method**.  
* ray\_tracer.py: Implement find\_nearest\_object(ray, objects).

### **Tasks**

1. **Sphere Math:** Solve the quadratic equation $|O \+ tD \- C|^2 \= R^2$. Return the smallest positive $t$.  
2. **Plane Math:** Solve $(P \- P\_0) \\cdot N \= 0$.  
3. **Cube Math:** Calculate intersection intervals $t\_{min}$ and $t\_{max}$ for X, Y, and Z axes. Check for interval overlap.  
4. **Nearest Neighbor:** Loop through all objects for a given ray, keeping track of the closest intersection (smallest positive $t$).

### **Measurable Testing**

1. **Silhouette Render:** In ray\_tracer.py, set pixel color to **White** if it hits any object, **Black** otherwise.  
2. **Visual Verification:** Render pool.txt. You should see the clear white shapes of balls, the triangle rack (cubes), and the floor plane against a black background.

## **Phase 3: Shading & Illumination (Phong Model)**

**Goal:** Calculate surface color using local illumination (Diffuse \+ Specular).

### **Directory Structure Updates**

* ray\_tracer.py: Implement calc\_color(object, intersection, lights, scene\_settings).

### **Tasks**

1. **Diffuse:** Calculate $K\_d \\times I\_L \\times (N \\cdot L)$.  
2. **Specular:** Calculate $K\_s \\times I\_L \\times (R \\cdot V)^\\alpha$ (where $\\alpha$ is shininess).  
3. **Ambient/Background:** Apply scene background color if no intersection occurs.

### **Measurable Testing**

1. **Matte Test:** Set specular intensity to 0 in the code. Render Spheres.png. Spheres should look like dull, shaded balls.  
2. **Shiny Test:** Re-enable specular. Verify bright white highlights appear on the balls facing the light sources.

## **Phase 4: Soft Shadows**

**Goal:** Implement realistic lighting occlusion with penumbra.

### **Directory Structure Updates**

* ray\_tracer.py: Implement compute\_shadow\_intensity(light, point, objects).

### **Tasks**

1. **Light Grid:** Define a rectangle on the light source perpendicular to the direction of the surface point.  
2. **Jittered Sampling:** Divide the light rectangle into $N \\times N$ cells (where $N$ is root\_number\_shadow\_rays). Pick a random point inside each cell.  
3. **Cast Shadow Rays:** Shoot a ray from the surface point to the sampled light point.  
4. **Intensity:** Accumulate light based on the percentage of rays that reach the light source unblocked.

### **Measurable Testing**

1. **Hard Shadow:** Set root\_number\_shadow\_rays \= 1 in pool.txt. Shadows should be pitch black with jagged edges.  
2. **Soft Shadow:** Set root\_number\_shadow\_rays \= 5\. Shadow edges should blend smoothly into the background.

## **Phase 5: Recursive Ray Tracing**

**Goal:** Handle global illumination effects like reflection and transparency.

### **Directory Structure Updates**

* ray\_tracer.py: Refactor the main logic into a recursive function trace\_ray(ray, recursion\_level).

### **Tasks**

1. **Reflection:** If object reflectivity $\> 0$, compute reflection vector $R \= V \- 2N(V \\cdot N)$. Recursively call trace\_ray(R, level+1).  
2. **Transparency:** If object transparency $\> 0$, cast a secondary ray through the object.  
3. **Combination:** Combine (1 \- transparency) \* (Diffuse \+ Specular) \+ Reflection \+ Transparency\_Color.  
4. **Base Case:** Return background color if recursion\_level equals max\_recursions.

### **Measurable Testing**

1. **Reflection Check:** Inspect the billiard balls in pool.png. You should see reflections of neighboring balls on their surfaces.  
2. **Mirror Plane:** Create a temporary test scene with a floor plane having reflection (1,1,1). It should act as a perfect mirror.

## **Phase 6: Bonus & Optimization**

**Goal:** Performance improvements and advanced features.

### **Directory Structure Updates**

* ray\_tracer.py: Refactor heavy loops to use NumPy vectorization.

### **Tasks**

1. **Transparent Shadows:** Modify shadow logic to allow partial light through transparent objects (Bonus).  
2. **Vectorization:** Instead of processing pixel-by-pixel in Python loops, try to process rows or the entire image matrix using NumPy operations.

### **Measurable Testing**

1. **Speed Test:** Measure execution time before and after vectorization.  
2. **Visual Match:** Ensure the final output matches the provided example pool.png nearly pixel-perfectly.