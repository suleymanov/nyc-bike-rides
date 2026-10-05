import requests


SERVICE_URL = 'http://127.0.0.1:8001/predict'


def test_predict_local():

    samples = [
        ('E 138 St & Cypress Ave', 0.0, 0.0, 0.0, 3, 0, 2, 0),
        ('W 43 St & 6 Ave', 2.0, 8.0, 16.0, 3, 1, 14, 0),
        ('Mercer St & Bleecker St', 2.0, 3.0, 3.0, 3, 2, 23, 0),
        ('Clinton St & Joralemon St', 3.0, 3.0, 7.0, 3, 5, 18, 0),
        ('Cliff St & Fulton St', 0.0, 0.0, 0.0, 3, 1, 5, 1)
    ]
    answers = (0, 3, 1, 2, 1)

    for (esn, c30, c60, c120, m, wd, h, hp), ans in zip(samples, answers):

        response = requests.post(
            SERVICE_URL,
            json={
                'end_station_name': esn,
                'count_30': c30,
                'count_60': c60,
                'count_120': c120,
                'month': m,
                'weekday': wd,
                'hour': h,
                'hour_part': hp
            }
        )
        
        prediction = response.json()
        assert response.status_code == 200
        assert prediction == ans


if __name__ == '__main__':
    test_predict_local()
