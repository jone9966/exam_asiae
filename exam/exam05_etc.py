if __name__ == "__main__" :
    a = [1, 2, 3]
    b = 4

    try :
        print('before')
        c = b / a[3]
        print('after')
    except IndexError :
        print('인덱스가 없습니다.')
    finally :
        print(c)