

cosumer_detail = '/retail-us/customer-read/v1/consumers?userName={}' #it will receive the cosumer detail with consumer id
consumer_account_list = '/retail-us/account/v1/consumers/{}/account-order' #it will retreive the consumer accounts
consumer_account_type = '/retail-us/account/v1/consumers/832/accounts' #it will retreive the consumer account type (current, saving, loan)
consumer_account_extended_details = '/retail-us/account/v1/consumers/832/accounts/extended' #it will retreive the extended details like (int allowed or not with balances and actual account number)
consumer_single_account_details = '/retail-us/account/v1/consumers/831/accounts/10020' #it will retreive single account extended details
consumer_single_account_balances = '/retail-us/account/v1/consumers/831/accounts/10020/details' #it will show extended balance information
consumer_account_transaction = '/retail-us/account/v1/consumers/831/accounts/10020/transactions?offset=0&amp; limit=100'
beneficiary_list = '/corporate/beneficiary-maintenance/me/v1/beneficiaries'
