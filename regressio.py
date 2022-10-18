from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def mallinna(malli, X, y):
    
    '''Jakaa datan testi- ja opetusdataan, sovittaa mallin, tulostaa selityskertoimet,
    tulostaa opetusdatan virhetermit kaaviona sekä tulostaa toteutuneet ja ennusteet hajontakaaviona.'''
    
    # Jako opetus- ja testidataan
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
    
    # Mallin sovitus opetusdataan
    malli.fit(X_train, y_train)
            
    # Selityskerroin opetusdatalle
    y_pred_train = malli.predict(X_train)
    R2_train_malli = malli.score(X_train, y_train)
        
    # Selityskerroin testidatalle
    y_pred_test = malli.predict(X_test)
    R2_test_malli = malli.score(X_test, y_test)
        
    # Selityskertoimien tulostus
    print(f'Opetusdatan selityskerroin {R2_train_malli:.3f}')
    print(f'Testidatan selityskerroin {R2_test_malli:.3f}')
       
    # Opetusdatan virhetermit kaaviona
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].set_title('Ennustevirheiden jakauma opetusdatassa')
    sns.histplot((y_train-y_pred_train), kde=True, ax=ax[0])
    ax[0].set_xlabel('y_train - y_pred_train')
    
    # toteutuneet ja ennustetut hajontakaaviona testidatalle
    ax[1].set_title('Toteutuneet ja ennustetut testidatassa')
    ax[1].scatter(x=y_test, y=y_pred_test)
    ax[1].set_xlabel('toteutunut')
    ax[1].set_ylabel('ennuste')

