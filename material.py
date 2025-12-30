class Material:
    def __init__(self, diffuse_color, specular_color, reflection_color, shininess, transparency):
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.reflection_color = reflection_color
        self.shininess = shininess
        self.transparency = transparency

    def blend_colors(self, local_color, reflection_color, transparency_color):
        """
        Combine local, reflection, and transparency colors using the PDF formula.

        Formula (PDF page 5):
            output = (background) · transparency
                   + (diffuse + specular) · (1 − transparency)
                   + (reflection)

        Args:
            local_color: Diffuse + specular from Phong shading (numpy array)
            reflection_color: Color from reflections (numpy array)
            transparency_color: Color from transparency (background) (numpy array)

        Returns:
            numpy array: Final combined RGB color [0, 1]
        """
        return (
            transparency_color +                      # background · transparency
            local_color * (1.0 - self.transparency) + # local · (1 - transparency)
            reflection_color                          # reflection (independent!)
        )
