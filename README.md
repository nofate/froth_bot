Please scroll the page down if you don't understand Russian and yet want to know what's going on here :)

# Куда я попал?

В этом репозитории лежит код решения задачи с [хакатона Норникеля](nnhackathon.ru) (трек "Пенная вечеринка"), команда "Свидетели редких природных явлений".

## Вход
56 коротких видео с металлургического производства, с этапа флотации [1].

## Выход

### 1) Статика: пузыри и их характеристики

С помощью бинаризации мы нашли блики, затем аппроксимировали пузыри эллипсами и придумали, как считать ряд величин, характеризующих состояние пузырьковой смеси в бочке в конкретный момент.

Вот как выглядит результат работы алгоритма:

![static-features](https://github.com/nofate/froth_bot/blob/master/assets/bubbles1.gif)


### 2) Динамика: скорости пузырьков и направление потока

Для решения этой части задачи мы использовали SIFT-дескрипторы и особые точки, затем сложным алгоритмом сматчили особые точки из разных фреймов видео между собой (сложность была в том, чтобы минимизировать число ложноположительных срабатываний алгоритма матчинга), получили сдвиги между парами точек и придумали, как с их помощью считать скорости пузырей в пене.
Еще мы научились определять, в каком направлении движется поток пузырей.

Результат работы этой части алгоритма можно посмотреть ниже:

![dynamic-features](https://github.com/nofate/froth_bot/blob/master/assets/bubbles2.gif)

Цветными линиями показаны смещения сопоставленных пар особых точек SIFT, цвет линии зависит от направления, в котором движется пузырь (в качестве направлений брали движение под разными углами с шагом 45 градусов).


# Where am I? What is this?

Here you can find our team solution for [Nornickel's hackathon](nnhackathon.ru) 2nd task. The goal was to augment froth flotation process, to make it more controllable.

## Input

As an input we had 56 1-minute recordings with froth flotation. No annotation was provided.

## Output

We've used bunch of classical CV methods: from constant thresholding to SIFT descriptors, from `import cv2` to home-made algorithms to suppress false-positives.

### 1) Static
To measure properties of bubbles in froth, we've used combination of constant thresholding and Sobel filter, after that we've implemented custom algorithm to find circumscribing ellipse for the bubble (if it is possible).

### 2) Dynamic
To extract speed we've implemented simplified version of algorithm which was given at [2]. We've extracted SIFT keypoints and descriptors, after that we've matched keypoints at each pair of video frames and used this to compute keypoints "pseudo-speeds".

# References
[1]: https://ru.wikipedia.org/wiki/%D0%A4%D0%BB%D0%BE%D1%82%D0%B0%D1%86%D0%B8%D1%8F
[2]: Jinping LIU, Weihua GUI, Zhaohui TANG: Flow velocity measurement and analysis based on froth image SIFT features and Kalman filter for froth flotation. Turk J Elec Eng & Comp Sci, (2013) 21: 2378 – 2396. doi:10.3906/elk-1204-91
