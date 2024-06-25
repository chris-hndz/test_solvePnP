using UnityEngine;

public class CameraSetup : MonoBehaviour
{
    public float fx = 775.9587; // Longitud focal en x
    public float fy = 775.9587; // Longitud focal en y
    public float cx = 448; // Punto principal en x
    public float cy = 448; // Punto principal en y
    public int imgWidth = 1080; // Ancho de la imagen
    public int imgHeight = 1920; // Alto de la imagen
    public Camera cam;
    void Start()
    {
        //Camera cam = Camera.main;

        // Calcular el FOV en la dirección vertical
        float fovY = 2 * Mathf.Atan(imgHeight / (2 * fy)) * Mathf.Rad2Deg;
        cam.fieldOfView = fovY;

        // Configurar el desplazamiento de la lente
        cam.lensShift = new Vector2((cx - imgWidth / 2) / imgWidth, (cy - imgHeight / 2) / imgHeight);
    }
}
