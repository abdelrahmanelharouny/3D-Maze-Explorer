#version 330 core

// Inputs from vertex shader
in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;

// Uniforms
uniform sampler2D texture1;
uniform vec3 viewPos;

struct DirLight {
    vec3 direction;
    float ambient;
    float diffuse;
    float specular;
};
uniform DirLight light;

// Output color
out vec4 FragColor;

void main()
{
    // Sample texture
    vec4 texColor = texture(texture1, TexCoord);
    
    // Normalize vectors
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(-light.direction);
    vec3 viewDir = normalize(viewPos - FragPos);
    
    // Ambient lighting
    vec3 ambient = light.ambient * vec3(texColor.rgb);
    
    // Diffuse lighting (Lambert)
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * vec3(texColor.rgb);
    
    // Specular lighting (Blinn-Phong)
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), 32.0);
    vec3 specular = light.specular * spec * vec3(1.0); // White specular
    
    // Combine lighting
    vec3 result = ambient + diffuse + specular;
    
    // Apply texture alpha
    FragColor = vec4(result, texColor.a);
}
