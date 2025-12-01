## Запуск решения 
* Клонируем репо: `git clone git@github.com:LucasAlegre/sumo-rl.git`  
* Копируем в директорию `sumo_rl/experiments` наши файлы: `cp ql_4x4grid_custom.py dqn_big_intersection_custom.py analyze_results.py sumo_rl/experiments`  
* Переходим в директорию репозитория `cd sumo_rl`  
* Ставим зависимости `sudo add-apt-repository ppa:sumo/stable && sudo apt-get update && sudo apt-get install sumo sumo-tools sumo-doc` и экспортируем `export SUMO_HOME="/usr/share/sumo"`  
* Создаем окружение и ставим зависимсти `pip install sumo-rl gymnasium torch numpy pandas`  
* Запускаем кастомный Q-learning на 4x4 сетке: `python experiments/ql_4x4grid_custom.py`  
* Запускаем кастомный DQN на большой развязке: `python experiments/dqn_big_intersection_custom.py`  
* Запускаем стандартный Q-learning на 4x4 сетке: `python experiments/ql_4x4grid.py`  
* Запускаем стандартный DQN на большой развязке: `python experiments/dqn_big-intersection.py`  

## P.S.
Обучение прошло не очень хорошо, так как было мало вермени и не было GPU - модели просто недообучились... Параметры с которыми запускал можно посмотреть в самих файлах с решением, параметры бейзлайнов - для QL схожие, для dqn стандартные
