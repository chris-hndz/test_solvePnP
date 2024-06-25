using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScreenPosition : MonoBehaviour
{
    public Camera cam;  
    public List<Transform> objectList = new List<Transform>();
    private List<Vector3> screenPointList = new List<Vector3>();
    
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        // Presionar "J" para activar
        if (Input.GetKeyDown(KeyCode.J) && objectList.Count > 0)
        {
            screenPointList.Clear();
            for (int i = 0; i < objectList.Count; i++){
                Vector3 screenPos = cam.WorldToScreenPoint(objectList[i].position);
                screenPointList.Add(screenPos);
                Debug.Log($"Cubo {i}: Posicion 3D {objectList[i].position} -> Posicion en pantalla {screenPointList[i]}");
            }
        }
    }
}
