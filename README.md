## Классификация возрастной группы людей по изображению

# Формулировка задачи

Целью проекта является разработка модели для определения возраста человека по его чертам
лица на изображениях. Эта задача имеет различные области применения. Например, она может
быть использована рекламными компаниями для персонализации рекламных объявлений в
зависимости от возрастной группы аудитории или для фильтрации контента в соответствии с
возрастными ограничениями.

# Данные

Данные будут взяты с https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset
.

Набор данных Indian Movie Face database (IMFDB) содержит 19906 изображений лиц, собранных
со 100 индийских актеров из более чем 100 видеороликов. В этом наборе данных представлена
высокая степень изменчивости в масштабе, позе, выражении, освещении, возрасте, разрешении,
заслонении и макияже. Благодаря большой вариативности, я думаю, этих данных должно быть
достаточно для получения хороших результатов. Возможно, из-за выборки, состоящей только из
жителей Индии, на другие расы и национальности результат обобщаться будет хуже, однако
хотя бы для оценки возраста индийцев данных должно хватить. Возможно, сложности также
могут возникнуть из-за различных размеров изображений.

Каждому изображению присвоен один из трех классов: Young, Middle, Old. Так что задача
сводится к задаче мультиклассовой классификации.

# Подход к моделированию

В процессе работы планируется использование библиотеки pytorch. Модель будет состоять из
из слоев convolutional и max-pooling. Сверточные слои позволят извлекать важные признаки
из изображений лиц, в то время как слои max-pooling будут выполнять пространственное
уменьшение размерности данных, сохраняя важные признаки.

В качестве функции активации на выходном слое планируется использовать softmax для
предсказания вероятностей принадлежности к каждой возрастной группе.

Примерная схема представлена на рисунке ниже, но скорее всего будет доработана:
![image](https://github.com/lodochnikova/MLOps-project-age-classification/assets/72004887/ea7f53ba-9693-4e6c-bd8c-bdf88e7364cf)

# Способ предсказания

После обучения модели необходимо будет разработать пайплайн для предсказания возраста на
новых изображениях лиц.

Этот пайплайн будет включать в себя следующие шаги:

1. Предобработка изображений:

    Необходимо привести изображение к фиксированному размеру.

2. Обучение модели на предобработанных данных.

3. Оценка эффективности модели на валидационной выборке.

    В случае недостижения достаточно хороших результатов следует изменить архитектуру
    модели и попробовать новые подходы.

4. Тестирование модели на тестовой выборке для оценки ее способностей к прогнозированию.

5. Применение модели на новых данных для определения возрастной группы человека по чертам
   его лица.
