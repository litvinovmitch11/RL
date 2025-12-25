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
    
    [Header("Границы арены")]
    public float arenaSize = 7.5f;

    [Header("Сенсоры")]
    public Unity.MLAgentsExamples.GroundContact[] groundContacts;

    private float distToTarget = 0f;
    private float bestDistance = 0f;
    private Vector3 lastPosition;
    private float timeWithoutProgress = 0f;
    private float episodeStartTime;
    private float[] previousActions;
    private int stepsInEpisode = 0;

    public override void Initialize()
    {
        distToTarget = Vector3.Distance(body.transform.position, cube.transform.position);
        defRot = body.transform.rotation;
        defPos = body.transform.position;
        bestDistance = distToTarget;
        lastPosition = body.transform.position;
        previousActions = new float[12];
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
            previousActions[i] = 0f;
        }
        
        stepsInEpisode = 0;
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
        
        // Случайная позиция цели в пределах арены, но не слишком близко к краю
        float safeMargin = arenaSize * 0.8f;
        cube.transform.position = new Vector3(
            Random.Range(-safeMargin, safeMargin), 
            0.21f, 
            Random.Range(-safeMargin, safeMargin)
        );
        
        distToTarget = Vector3.Distance(body.transform.position, cube.transform.position);
        bestDistance = distToTarget;
        lastPosition = body.transform.position;
        timeWithoutProgress = 0f;
        episodeStartTime = Time.time;
        stepsInEpisode = 0;
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
        stepsInEpisode++;
        
        // Сохраняем предыдущие действия для плавности
        float[] currentActions = new float[12];
        for (int i = 0; i < 12; i++)
        {
            currentActions[i] = actions[i];
        }
        
        // Применяем действия к ногам с плавностью
        for (int i = 0; i < 12; i++)
        {
            // Плавность движений: смешиваем с предыдущим действием
            float smoothedAction = Mathf.Lerp(previousActions[i], actions[i], 0.4f);
            float angle = Mathf.Lerp(legs[i].xDrive.lowerLimit, legs[i].xDrive.upperLimit, 
                                    (smoothedAction + 1) * 0.5f);
            MoveLeg(legs[i], angle);
            previousActions[i] = smoothedAction;
        }
        
        // Рассчитываем текущее расстояние до цели
        float currentDistance = Vector3.Distance(body.transform.position, cube.transform.position);
        
        // 1. Основная награда за приближение к цели
        float distanceImprovement = distToTarget - currentDistance;
        if (distanceImprovement > 0.01f)
        {
            // Большая награда за приближение
            AddReward(distanceImprovement * 0.8f);
            bestDistance = Mathf.Min(bestDistance, currentDistance);
            timeWithoutProgress = 0f;
        }
        else if (distanceImprovement < -0.01f)
        {
            // Небольшой штраф за отдаление
            AddReward(distanceImprovement * 0.3f);
            timeWithoutProgress += Time.fixedDeltaTime;
        }
        else
        {
            // Очень маленький штраф за отсутствие движения
            AddReward(-0.001f);
            timeWithoutProgress += Time.fixedDeltaTime;
        }
        
        // 2. Награда за ориентацию к цели головой вперед
        Vector3 toTarget = (cube.transform.position - body.transform.position).normalized;
        float forwardDot = Vector3.Dot(body.transform.right.normalized, toTarget);
        
        // Больше награды за точное направление к цели
        if (forwardDot > 0.7f)
        {
            AddReward(forwardDot * 0.1f);
        }
        else if (forwardDot > 0.3f)
        {
            AddReward(forwardDot * 0.05f);
        }
        
        // 3. Награда за движение в правильном направлении
        if (body.velocity.magnitude > 0.1f)
        {
            float velocityAlignment = Vector3.Dot(body.velocity.normalized, toTarget);
            if (velocityAlignment > 0)
            {
                AddReward(velocityAlignment * 0.08f);
            }
        }
        
        // 4. Награда за стабильность (меньше падений/колебаний)
        // Награда за правильную высоту
        float heightReward = Mathf.Exp(-Mathf.Abs(body.transform.position.y + 0.5f) * 3f);
        AddReward(heightReward * 0.02f);
        
        // Награда за горизонтальную ориентацию
        float upDot = Vector3.Dot(body.transform.up, Vector3.up);
        AddReward(Mathf.Clamp(upDot, 0f, 1f) * 0.03f);
        
        // Штраф за слишком сильные колебания
        float angularPenalty = -body.angularVelocity.magnitude * 0.01f;
        AddReward(angularPenalty);
        
        // 5. Награда за правильную походку (все 4 ноги на земле)
        int groundContactCount = 0;
        foreach(var contact in groundContacts)
        {
            if(contact.touchingGround) groundContactCount++;
        }
        
        // Награда за каждую ногу на земле
        AddReward(groundContactCount * 0.01f);
        
        // Большая награда за все 4 ноги на земле
        if (groundContactCount >= 4)
        {
            AddReward(0.02f);
        }
        
        // Штраф за прыжки (все ноги в воздухе)
        if (groundContactCount == 0)
        {
            AddReward(-0.05f);
        }
        
        // 6. Большая награда за достижение цели
        if (currentDistance < 0.5f)
        {
            // Дополнительный бонус за ориентацию к цели при достижении
            float finalAlignment = Vector3.Dot(body.transform.right.normalized, toTarget);
            float alignmentBonus = Mathf.Clamp(finalAlignment, 0f, 1f) * 3f;
            
            // Бонус за скорость достижения
            float timeBonus = Mathf.Clamp(1f / (stepsInEpisode * 0.02f + 1f), 0f, 1f) * 4f;
            
            AddReward(15f + alignmentBonus + timeBonus);
            EndEpisode();
            return;
        }
        
        // 7. Штраф за падение
        if (body.transform.position.y < -0.5f)
        {
            AddReward(-8f);
            EndEpisode();
            return;
        }
        
        // 8. Штраф за приближение к границам арены
        float distanceToEdgeX = arenaSize - Mathf.Abs(body.transform.position.x);
        float distanceToEdgeZ = arenaSize - Mathf.Abs(body.transform.position.z);
        float minDistanceToEdge = Mathf.Min(distanceToEdgeX, distanceToEdgeZ);
        
        if (minDistanceToEdge < 1.5f)
        {
            // Увеличиваем штраф ближе к краю
            AddReward(-(1.5f - minDistanceToEdge) * 0.3f);
        }
        
        // 9. Штраф за выход за пределы арены
        if (Mathf.Abs(body.transform.position.x) > arenaSize || 
            Mathf.Abs(body.transform.position.z) > arenaSize)
        {
            AddReward(-6f);
            EndEpisode();
            return;
        }
        
        // 10. Штраф за застревание (нет прогресса 6 секунд)
        if (timeWithoutProgress > 6f)
        {
            AddReward(-3f);
            EndEpisode();
            return;
        }
        
        // 11. Штраф за слишком долгое выполнение эпизода
        if (stepsInEpisode > 2500)
        {
            AddReward(-2f);
            EndEpisode();
            return;
        }
        
        // 12. Небольшой штраф за каждый шаг (поощряем эффективность)
        AddReward(-0.0005f);
        
        // 13. Штраф за слишком резкие движения ног
        float actionChangePenalty = 0f;
        for (int i = 0; i < 12; i++)
        {
            actionChangePenalty += Mathf.Abs(actions[i] - previousActions[i]);
        }
        AddReward(-actionChangePenalty * 0.0005f);
        
        // 14. Награда за плавное движение вперед
        float forwardMovement = Vector3.Dot(body.velocity, body.transform.right);
        if (forwardMovement > 0.2f && groundContactCount >= 3)
        {
            AddReward(forwardMovement * 0.01f);
        }
        
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
        
        // Визуализация границ арены
        Debug.DrawLine(new Vector3(-arenaSize, 0, -arenaSize), new Vector3(arenaSize, 0, -arenaSize), Color.blue);
        Debug.DrawLine(new Vector3(arenaSize, 0, -arenaSize), new Vector3(arenaSize, 0, arenaSize), Color.blue);
        Debug.DrawLine(new Vector3(arenaSize, 0, arenaSize), new Vector3(-arenaSize, 0, arenaSize), Color.blue);
        Debug.DrawLine(new Vector3(-arenaSize, 0, arenaSize), new Vector3(-arenaSize, 0, -arenaSize), Color.blue);
    }

    void MoveLeg(ArticulationBody leg, float targetAngle)
    {
        leg.GetComponent<Leg>().MoveLeg(targetAngle, servoSpeed);
    }
    
    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject == cube)
        {
            Debug.Log("Достигнута цель!");
        }
    }
}
