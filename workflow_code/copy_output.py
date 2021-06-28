from datetime import date
import paramiko


def server_login():
    """logs into the BW-HPC server

    Returns:
        client: returns paramiko client
    """
    hostname = "134.2.168.52"
    username = "sgutwein84"
    password = "Derzauberkoenig1!"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=hostname, username=username, password=password)

    return client


if __name__ == "__main__":

    path = "/work/ws/nemo/tu_zxoys08-EGS-0/egs_home/dosxyznrc/output"

    client = server_login()

    d = date.today()
    if int(d.month) < 10:
        month = "0" + str(d.month)
    else:
        month = str(d.month)

    if int(d.day) < 10:
        day = "0" + str(d.day)
    else:
        day = str(d.day)

    with open("/home/baumgartner/sgutwein84/copy_output.sh", "w") as fout:

        fout.write(
            f"scp -r tu_zxoys08@login1.nemo.uni-freiburg.de:{path} /home/baumgartner/sgutwein84/container/output_{d.year}{month}{day}"
        )

    _, stdout, _ = client.exec_command(
        "./copy_output.sh"
    )
