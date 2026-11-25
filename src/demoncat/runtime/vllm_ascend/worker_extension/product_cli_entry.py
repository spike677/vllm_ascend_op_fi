"""Versioned reference CLI transport; no daemon, Worker or permit creation.

This is not the yellow project's demoncat.cli implementation. Its trusted
session daemon must explicitly implement this protocol and authenticate peers.
"""
import argparse
import json
import socket
import sys

SCHEMA = 'demoncat.product.cli.v1'
LIMIT = 1024 * 1024


def invoke(path, action, mechanism=None):
    request = dict(schema=SCHEMA, action=action)
    if mechanism is not None: request['mechanism'] = mechanism
    raw = (json.dumps(request, ensure_ascii=False) + '\n').encode('utf-8')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(1800)
        connection.connect(path)
        connection.sendall(raw)
        stream = connection.makefile('rb')
        response = stream.readline(LIMIT + 1)
    if len(response) > LIMIT or not response.endswith(b'\n'):
        raise RuntimeError('PRODUCT_CLI_RESPONSE_BOUND_EXCEEDED')
    result = json.loads(response)
    if (type(result) is not dict or result.get('schema') != SCHEMA
            or type(result.get('exit_code')) is not int or result['exit_code'] not in (0,4,5)):
        raise RuntimeError('PRODUCT_CLI_RESPONSE_SCHEMA_INVALID')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(prog='dcatpfi-product')
    commands=parser.add_subparsers(dest='command', required=True)
    fault=commands.add_parser('fault')
    actions=fault.add_subparsers(dest='action', required=True)
    for action in ('list','enable','status','disable'):
        item=actions.add_parser(action); item.add_argument('--socket', required=True)
        if action=='enable': item.add_argument('--fault-id', required=True)
    args=parser.parse_args(argv)
    try: result=invoke(args.socket,args.action,getattr(args,'fault_id',None))
    except (OSError,ValueError,RuntimeError) as error:
        print(json.dumps(dict(error=str(error)),ensure_ascii=False),file=sys.stderr); return 5
    print(json.dumps(result,ensure_ascii=False));return result['exit_code']

if __name__=='__main__':raise SystemExit(main())
