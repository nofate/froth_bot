Please scroll the page down if you don't understand Russian and yet want to know what's going on here :)

# Куда я попал?

В этом репозитории лежит код решения задачи с хакатона Норникеля (трек "Пенная вечеринка"), команда "Свидетели редких природных явлений".

## Вход
56 коротких видео с металлургического производства, с этапа [флотации](https://ru.wikipedia.org/wiki/%D0%A4%D0%BB%D0%BE%D1%82%D0%B0%D1%86%D0%B8%D1%8F).

## Выход

### Статика: пузыри и их характеристики

С помощью бинаризации мы нашли блики, затем аппроксимировали пузыри эллипсами и придумали, как считать ряд величин, характеризующих состояние пузырьковой смеси в бочке в конкретный момент.

Вот как выглядит результат работы алгоритма:

![static-features](https://github.com/nofate/froth_bot/blob/master/assets/bubbles1.gif)


### Динамика: движение

Для решения этой части задачи мы использовали SIFT-дескрипторы и особые точки, затем сложным алгоритмом сматчили особые точки из разных фреймов видео между собой (сложность была в том, чтобы минимизировать число ложноположительных срабатываний алгоритма матчинга), получили сдвиги между парами точек и придумали, как с их помощью считать скорости пузырей в пене.
Еще мы научились определять, в каком направлении движется поток пузырей.

Результат работы этой части алгоритма можно посмотреть ниже:

![dynamic-features](https://github.com/nofate/froth_bot/blob/master/assets/bubbles2.gif)

# Where am I? What is this?

TBA

## Input

## Output

### Static

### Dynamic