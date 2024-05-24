import math

def calculate_angle(A, B, C):
    # Hitung vektor AC
    vec_AC = (C[0] - A[0], C[1] - A[1])
    # Hitung vektor AB
    vec_AB = (B[0] - A[0], B[1] - A[1])

    # Hitung dot product dari vektor AC dan AB
    dot_product = vec_AC[0] * vec_AB[0] + vec_AC[1] * vec_AB[1]

    # Hitung magnitudo dari vektor AC dan AB
    mag_AC = math.sqrt(vec_AC[0]**2 + vec_AC[1]**2)
    mag_AB = math.sqrt(vec_AB[0]**2 + vec_AB[1]**2)

    # Hitung kosinus sudut
    cos_angle = dot_product / (mag_AC * mag_AB)

    # Hitung sudut dalam radian
    angle_rad = math.acos(cos_angle)

    # Konversi sudut dari radian ke derajat
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def main():
    # Koordinat titik-titik
    A = (4, -1)
    B = (5, 1)
    C = (0, 0)

    # Hitung sudut antara titik-titik
    angle = calculate_angle(A, B, C)

    # Cek apakah sudut lebih dari 90 derajat atau tidak
    if angle > 90:
        print("Keluar")

    else:
        

        print("Masuk")





if __name__ == "__main__":
    main()
