#version 330 core

// Vertex attributes
layout (location = 0) in vec3 aPos;      // Vertex position
layout (location = 1) in vec2 aTexCoord; // Texture coordinates
layout (location = 2) in vec3 aNormal;   // Vertex normal

// Uniforms
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// Outputs to fragment shader
out vec2 TexCoord;
out vec3 FragPos;
out vec3 Normal;

void main()
{
    // Transform vertex to world space
    FragPos = vec3(model * vec4(aPos, 1.0));
    
    // Transform normal to world space
    Normal = mat3(transpose(inverse(model))) * aNormal;
    
    // Pass texture coordinates
    TexCoord = aTexCoord;
    
    // Final position in clip space
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
