from django.shortcuts import render
import requests
import json
from django.http import HttpResponse
from finbot import settings
from bot import api_url
import os
from . import helperFunction
import re
from requests_toolbelt.utils import dump


strong = settings.strong
client_id = settings.client_id
access_key = settings.access_key
token_endpoint = settings.token_endpoint
base_url = settings.base_url


# Create your views here.

def auth(request):
    # return render_template('index.html', strong=settings.strong)
    return render(request,'bot/index.html')

# Log in
def login(request):

    data = {'grant_type':'client_credentials'}
    if(strong):
        data['client_assertion_type'] = 'urn:ietf:params:oauth:client-assertion-type:jwt-bearer'
        data['client_assertion'] = helpers.jwToken()
    else:
        data['client_id'] = client_id
        data['client_secret'] = access_key

    headers = {'Content-Type':'application/x-www-form-urlencoded'}
    r = requests.post(token_endpoint, headers=headers, data=data)
    response =  r.json()

    if (r.status_code is 200):
        token = response['access_token']
        # Put token in the session
        request.session['session_token'] = token
        print("token set successfully")

    return render(request,'bot/assistant.html', { 'token': token, 'strong':strong})

# Display the results

def results(request):
    try:
        headers={"Authorization": "Bearer " + request.session['session_token']}
        result = requests.get(base_url + '/referential/v1/countries', headers=headers)
        print("before json_data")
        json_data = json.loads(result.text)['countries']
        print("after json_data")
        return render(request,'bot/results.html', {'results':json_data, 'strong':strong})
    except:
        return render(request,'bot/error.html', {'error':'Unauthorized!', 'strong':strong})


# Logout
def logout():
    session['session_token']=''
    return render_template('logout.html', error="You successfully removed the access token.", strong=strong)



def account_balance(request):
    try:
        headers={"Authorization": "Bearer " + request.session['session_token']}
        result = requests.get(base_url + '/retail-us/account/v1/consumers/831/accounts/10020/balances', headers=headers)
        print("before json_data")
        json_data = json.loads(result.text)
        print("after json_data {}".format(json_data))
        return render(request,'bot/account_balance.html', {'results':json_data, 'strong':strong})
    except:
        return render(request,'bot/error.html', {'error':'Unauthorized!', 'strong':strong})


def assistant(request):
    return render(request,'bot/assistant.html',{})

def apiCall(request):
    availableType = ['balance','transaction','post_id']
    text = request.GET.get('text')
    response  = requests.get('http://127.0.0.1:9999/model/?sentence='+text)

    # pred = helperFunction.get_pred(text)
    # print(pred)
    # flag = [thing for thing in availableType]
    # tt=[True for i in availableType if i in request.GET]
    # request_id=''.join([(i) for i in availableType if i in request.GET])
    # if tt:
    request_id = ""
    response = response.json()
    # decoded_response = json.loads(response)
    respone = response['intent']
    print(str(response['intent']).lower())
    balType=''
    if str(response['intent']).lower() == 'balance':
        acc_amount = re.findall(r'\b\d+\b', text)
        print("Acc = {}".format(acc_amount))
        if acc_amount:
            resp = prepareHeader('/retail-us/account/v1/consumers/831/accounts/'+acc_amount[0]+'/balances',request)
            resp = resp
            print(resp)
            for dd in resp:
                print(dd['type'])
                balType += '<br />' + dd['type'].capitalize()+ " " +str(dd['amount'])
            # for data in resp.json():
                # balType = ''.join(data['type']+data['amount'])
            finalResponse = "Here is your account balance: " +balType
        else:
            finalResponse = "Please let me know the account number"

    elif str(response['intent']).lower() == 'maketransaction':
        acc_amount = re.findall(r'\b\d+\b', text)

        internalTransfer = {
  "fromAccountId": "0001000002001",
  "toAccountId": "0001000003001",
  "amount": {
    "amount": "100",
    "currency": "USD"
  },
  "narrative": "FINBOT TNX"
}
        api_call_url = "/retail-banking/payments/v1/fund-transfers/internal"
        result = initiatePayment(api_call_url,request,internalTransfer)
        print(result['transactionId'])

        finalResponse = "Internal Transfer Executed Successfully, Here is the transaction details: <br />"
        finalResponse += "From Account: <b>0001000002001</b>"
        finalResponse += "<br />To Account: <b>{}</b>".format(str(acc_amount[0]))
        finalResponse += "<br /> Amount: <b>USD 100</b>"
        finalResponse += "<br />Transaction ID: <b>"+result['transactionId']+"</b>"
        finalResponse += "<br />Transaction Date: <b>"+result['transactionDate']+"</b>"
        finalResponse += "<br />Transaction Time: <b>"+result['transactionTime']+"</b>"
    else:

        resp = prepareHeader(api_url.beneficiary_list,request)
        print(resp)
    print(finalResponse)

    return HttpResponse(finalResponse)


def prepareHeader(api_call_url, request):
    # $('.msg_card_body').append('
    headers={"Authorization": "Bearer " + request.session['session_token']}
    result = requests.get(base_url + api_call_url, headers=headers)
    json_data = json.loads(result.text)
    return json_data

def initiatePayment(api_call_url, request, payload):
    headers={'Content-type': 'application/json',"Authorization": "Bearer " + request.session['session_token']}
    result = requests.post(base_url + api_call_url, headers=headers, json = payload).json()
    # prepared = result.prepare()
    # dump.dump_all(result)
    # print("{}".format(result))
    # json_data = json.loads(result.text)
    return result
