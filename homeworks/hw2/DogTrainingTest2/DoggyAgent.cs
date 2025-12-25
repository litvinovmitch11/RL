using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections;
using System;
using Random = UnityEngine.Random;
using UnityEngine.InputSystem;

public class DoggyAgent : Agent
{
    [Header("Сервоприводы")]
    public ArticulationBody[] legs;

    [Header("Скорость работы сервоприводов")]
    public float servoSpeed;

    [Header("Тело")]
    public ArticulationBody body;
    private Vector3 defPos;
    private Quaternion defRot;
    public float strenghtMove;

    [Header("Куб (цель)")]
    public GameObject cube;

    [Header("Сенсоры")]
    public Unity.MLAgentsExamples.GroundContact[] groundContacts;

    private float distToTarget = 0f;
    private float bestDistance = 0f;
    private Vector3 lastPosition;
    private float timeWithoutProgress = 0f;
    private float episodeStartTime;

    public override void Initialize()
    {
        distToTarget = Vector3.Distance(body.transform.position, cube.transform.position);
        defRot = body.transform.rotation;
        defPos = body.transform.position;
        bestDistance = distToTarget;
        lastPosition = body.transform.position;
    }

    public void ResetDog()
    {
        Quaternion newRot = Quaternion.Euler(-90, 0, Random.Range(0f, 360f));
        body.TeleportRoot(defPos, newRot);
        body.velocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;

        for (int i = 0; i < 12; i++)
        {
            MoveLeg(legs[i], 0);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        
        // Простая эвристика: синусоидальные движения для ходьбы
        float time = Time.time;
        for (int i = 0; i < 12; i++)
        {
            // Разные фазы для разных ног для походки
            float phase = (i % 4) * 0.25f;
            continuousActions[i] = Mathf.Sin(time * 2f + phase * Mathf.PI * 2f);
        }
    }

    public override void OnEpisodeBegin()
    {
        ResetDog();
        
        // Случайная позиция цели в пределах арены
        cube.transform.position = new Vector3(Random.Range(-7.5f, 7.5f), 0.21f, Random.Range(-7.5f, 7.5f));
        
        distToTarget = Vector3.Distance(body.transform.position, cube.transform.position);
        bestDistance = distToTarget;
        lastPosition = body.transform.position;
        timeWithoutProgress = 0f;
        episodeStartTime = Time.time;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 1. Позиция тела
        sensor.AddObservation(body.transform.position);
        
        // 2. Скорости тела
        sensor.AddObservation(body.velocity);
        sensor.AddObservation(body.angularVelocity);
        
        // 3. Направление "вперед" тела (право = вперед)
        sensor.AddObservation(body.transform.right);
        
        // 4. Абсолютная позиция цели
        sensor.AddObservation(cube.transform.position);
        
        // 5. Относительное положение цели
        Vector3 relativePosition = cube.transform.position - body.transform.position;
        sensor.AddObservation(relativePosition);
        
        // 6. Угол к цели относительно направления тела
        Vector3 toCube = (cube.transform.position - body.transform.position).normalized;
        float angleToCube = Vector3.SignedAngle(body.transform.right, toCube, Vector3.up);
        sensor.AddObservation(angleToCube);
        
        // 7. Расстояние до цели
        float distanceToCube = Vector3.Distance(body.transform.position, cube.transform.position);
        sensor.AddObservation(distanceToCube);
        
        // 8. Данные по каждой ноге
        foreach (var leg in legs)
        {
            // Целевой угол сервопривода
            sensor.AddObservation(leg.xDrive.target);
            // Линейная скорость ноги
            sensor.AddObservation(leg.velocity);
            // Угловая скорость ноги
            sensor.AddObservation(leg.angularVelocity);
        }
        
        // 9. Данные сенсоров касания земли
        foreach(var groundContact in groundContacts)
        {
            sensor.AddObservation(groundContact.touchingGround);
        }        
    }

    public override void OnActionReceived(ActionBuffers vectorAction)
    {
        var actions = vectorAction.ContinuousActions;
        
        // Применяем действия к ногам
        for (int i = 0; i < 12; i++)
        {
            float angle = Mathf.Lerp(legs[i].xDrive.lowerLimit, legs[i].xDrive.upperLimit, 
                                    (actions[i] + 1) * 0.5f);
            MoveLeg(legs[i], angle);
        }
        
        // Рассчитываем текущее расстояние до цели
        float currentDistance = Vector3.Distance(body.transform.position, cube.transform.position);
        
        // 1. Основная награда за приближение к цели
        float distanceImprovement = distToTarget - currentDistance;
        if (distanceImprovement > 0)
        {
            // Большая награда за приближение
            AddReward(distanceImprovement * 0.5f);
            bestDistance = Mathf.Min(bestDistance, currentDistance);
            timeWithoutProgress = 0f;
        }
        else
        {
            // Небольшой штраф за отдаление
            AddReward(distanceImprovement * 0.1f);
            timeWithoutProgress += Time.fixedDeltaTime;
        }
        
        // 2. Награда за движение в правильном направлении
        Vector3 toTarget = (cube.transform.position - body.transform.position).normalized;
        float forwardVelocity = Vector3.Dot(body.velocity.normalized, toTarget);
        AddReward(forwardVelocity * 0.05f);
        
        // 3. Награда за стабильность (меньше падений/колебаний)
        float heightReward = Mathf.Clamp(body.transform.position.y + 0.5f, 0f, 1f);
        AddReward(heightReward * 0.01f);
        
        // Штраф за слишком сильные колебания
        float angularPenalty = -body.angularVelocity.magnitude * 0.005f;
        AddReward(angularPenalty);
        
        // 4. Награда за контакт с землей (хотя бы 3 ноги на земле)
        int groundContactCount = 0;
        foreach(var contact in groundContacts)
        {
            if(contact.touchingGround) groundContactCount++;
        }
        if(groundContactCount >= 3)
        {
            AddReward(0.01f);
        }
        
        // 5. Большая награда за достижение цели
        if (currentDistance < 0.5f)
        {
            // Бонус за скорость достижения
            float timeBonus = Mathf.Clamp(1f / (Time.time - episodeStartTime + 1f), 0f, 1f);
            AddReward(10f + timeBonus * 5f);
            EndEpisode();
            return;
        }
        
        // 6. Штраф за падение
        if (body.transform.position.y < -0.5f)
        {
            AddReward(-5f);
            EndEpisode();
            return;
        }
        
        // 7. Штраф за выход за пределы арены
        if (Mathf.Abs(body.transform.position.x) > 9f || Mathf.Abs(body.transform.position.z) > 9f)
        {
            AddReward(-2f);
            EndEpisode();
            return;
        }
        
        // 8. Штраф за застревание (нет прогресса 5 секунд)
        if (timeWithoutProgress > 5f)
        {
            AddReward(-1f);
            EndEpisode();
            return;
        }
        
        // 9. Небольшой штраф за каждый шаг (поощряем эффективность)
        AddReward(-0.001f);
        
        // 10. Штраф за слишком резкие движения ног
        float actionSmoothness = 0f;
        for (int i = 0; i < 12; i++)
        {
            actionSmoothness += Mathf.Abs(actions[i]);
        }
        AddReward(-actionSmoothness * 0.0001f);
        
        // Обновляем расстояние для следующего шага
        distToTarget = currentDistance;
        lastPosition = body.transform.position;
    }
    
    public void FixedUpdate()
    {
        // Визуализация луча к цели
        Debug.DrawRay(body.transform.position, (cube.transform.position - body.transform.position).normalized * 2f, 
                     Color.green);
        Debug.DrawRay(body.transform.position, body.transform.right * 2f, Color.red);
    }

    void MoveLeg(ArticulationBody leg, float targetAngle)
    {
        leg.GetComponent<Leg>().MoveLeg(targetAngle, servoSpeed);
    }
    
    // Дополнительный метод для отладки (не обязательно для обучения)
    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject == cube)
        {
            Debug.Log("Достигнута цель!");
        }
    }
}
