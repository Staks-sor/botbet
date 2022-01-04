import pandas as pd
import numpy as np
import collections


data = pd.read_csv("data/RPL.csv", encoding='cp1251', delimiter=';')
data.head()


RPL_2018_2019 = pd.read_csv('data/Team Name 2018 2019.csv', encoding='cp1251')

teamList = RPL_2018_2019['Team Name'].tolist()



deleteTeam = [x for x in pd.unique(data['Соперник']) if x not in teamList]
for name in deleteTeam:
    data = data[data['Команда'] != name]
    data = data[data['Соперник'] != name]
data = data.reset_index(drop=True)


# Функция, возвращающая статистику команды за сезон
def GetSeasonTeamStat(team, season):
    goalScored = 0  # Голов забито
    goalAllowed = 0  # Голов пропущено

    gameWin = 0  # Выиграно
    gameDraw = 0  # Ничья
    gameLost = 0  # Проиграно

    totalScore = 0  # Количество набранных очков

    matches = 0  # Количество сыгранных матчей

    xG = 0  # Ожидаемые голы

    shot = 0  # Удары
    shotOnTarget = 0  # Удары в створ

    cross = 0  # Навесы
    accurateCross = 0  # Точные навесы

    totalHandle = 0  # Владение мячом
    averageHandle = 0  # Среднее владение мячом за матч

    Pass = 0  # Пасы
    accuratePass = 0  # Точные пасы

    PPDA = 0  # Интенсивность прессинга в матче

    for i in range(len(data)):
        if (((data['Год'][i] == season) and (data['Команда'][i] == team) and (data['Часть'][i] == 2)) or (
                (data['Год'][i] == season - 1) and (data['Команда'][i] == team) and (data['Часть'][i] == 1))):
            matches += 1

            goalScored += data['Забито'][i]
            goalAllowed += data['Пропущено'][i]

            if (data['Забито'][i] > data['Пропущено'][i]):
                totalScore += 3
                gameWin += 1
            elif (data['Забито'][i] < data['Пропущено'][i]):
                gameLost += 1
            else:
                totalScore += 1
                gameDraw += 1

            xG += data['xG'][i]

            shot += data['Удары'][i]
            shotOnTarget += data['Удары в створ'][i]

            Pass += data['Передачи'][i]
            accuratePass += data['Точные передачи'][i]

            totalHandle += data['Владение'][i]

            cross += data['Навесы'][i]
            accurateCross += data['Точные навесы'][i]

            PPDA += data['PPDA'][i]

    averageHandle = round(totalHandle / matches, 3)  # Владение мячом в среднем за матч

    return [gameWin, gameDraw, gameLost,
            goalScored, goalAllowed, totalScore,
            round(xG, 3), round(PPDA, 3),
            shot, shotOnTarget,
            Pass, accuratePass,
            cross, accurateCross,
            round(averageHandle, 3)]


GetSeasonTeamStat("Спартак", 2018)  # Статистика Спартака за сезон 2017/2018

returnNames = ["Выиграно", "Ничья", "Проиграно",
               "\nГолов забито", "Голов пропущено", "\nНабрано очков",
               "\nxG (за сезон)", "PPDA (за сезон)",
               "\nУдары", "Удары в створ",
               "\nПасы", "Точные пасы",
               "\nНавесы", "Точные навесы",
               "\nВладение (в среднем за матч)"]

for i, n in zip(returnNames, GetSeasonTeamStat("Спартак", 2018)):
    print(i, n)







# Функция, возвращающая статистику всех команд за сезон
def GetSeasonAllTeamStat(season):
    annual = collections.defaultdict(list)
    for team in teamList:
        team_vector = GetSeasonTeamStat(team, season)
        annual[team] = team_vector
    return annual



# Получение тренировычных данных
def GetTrainingData(seasons):
    totalNumGames = 0
    for season in seasons:
        annual = data[data['Год'] == season]
        totalNumGames += len(annual.index)
    numFeatures = len(GetSeasonTeamStat('Зенит', 2016))  # случайная команда для определения размерности
    xTrain = np.zeros((totalNumGames, numFeatures))
    yTrain = np.zeros((totalNumGames))
    indexCounter = 0
    for season in seasons:
        team_vectors = GetSeasonAllTeamStat(season)
        annual = data[data['Год'] == season]
        numGamesInYear = len(annual.index)
        xTrainAnnual = np.zeros((numGamesInYear, numFeatures))
        yTrainAnnual = np.zeros((numGamesInYear))
        counter = 0
        for index, row in annual.iterrows():
            team = row['Команда']
            t_vector = team_vectors[team]
            rivals = row['Соперник']
            r_vector = team_vectors[rivals]

            diff = [a - b for a, b in zip(t_vector, r_vector)]

            if len(diff) != 0:
                xTrainAnnual[counter] = diff
            if team == row['Победитель']:
                yTrainAnnual[counter] = 1
            else:
                yTrainAnnual[counter] = 0
            counter += 1
        xTrain[indexCounter:numGamesInYear + indexCounter] = xTrainAnnual
        yTrain[indexCounter:numGamesInYear + indexCounter] = yTrainAnnual
        indexCounter += numGamesInYear
    return xTrain, yTrain


years = range(2016, 2019)
xTrain, yTrain = GetTrainingData(years)


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(xTrain, yTrain)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
                 normalize=False)


def createGamePrediction(team1_vector, team2_vector):
    diff = [[a - b for a, b in zip(team1_vector, team2_vector)]]
    predictions = model.predict(diff)
    return predictions


team1_name = "Зенит"
team2_name = "Спартак"

team1_vector = GetSeasonTeamStat(team1_name, 2019)
team2_vector = GetSeasonTeamStat(team2_name, 2019)

print('Вероятность, что выиграет ' + team1_name + ':', createGamePrediction(team1_vector, team2_vector))
print('Вероятность, что выиграет ' + team2_name + ':', createGamePrediction(team2_vector, team1_vector))

for team_name in teamList:
    team1_name = "ЦСКА"
    team2_name = team_name

    if (team1_name != team2_name):
        team1_vector = GetSeasonTeamStat(team1_name, 2019)
        team2_vector = GetSeasonTeamStat(team2_name, 2019)

        print(team1_name, createGamePrediction(team1_vector, team2_vector), " - ", team2_name,
              createGamePrediction(team2_vector, team1_vector, ))
