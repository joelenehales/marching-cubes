// Phong fragment shader
#version 400

// Interpolated values from the vertex shaders
in vec3 Normal_cameraspace;
in vec3 EyeDirection_cameraspace;
in vec3 LightDirection_cameraspace;

// Ouput data
out vec4 color;  // Color, with lighting

// Uniforms, stay constant for the whole mesh
uniform vec4 modelColor;  // Object's base color
uniform float shininess;

void main(){

    // Material properties
    vec4 diffuse = modelColor;
    vec4 ambient = vec4(0.2,0.2,0.2,1.0);
    vec4 specular = vec4(1.0, 1.0, 1.0, 1.0);
    float shininess = 64;

    // Compute color
    vec3 n = normalize( Normal_cameraspace );
    vec3 l = normalize( LightDirection_cameraspace );
    float cosTheta = clamp( dot( n,l ), 0,1 ); //ensure dot product is between 0 and 1

    vec3 E = normalize(EyeDirection_cameraspace);
    vec3 R = reflect(-l,n);
    float cosAlpha = clamp( dot( E,R ), 0,1 );

    color = ambient + diffuse*cosTheta + specular*pow(cosAlpha,shininess);
}