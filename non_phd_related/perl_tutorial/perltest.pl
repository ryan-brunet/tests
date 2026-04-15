#!/usr/bin/env perl
use strict;
use warnings;

print "Hello, World!\n";

my $var1 = 10;
my @array = (1, 2, 3, 4, 5);

print "var1 = $var1\n";
print "array = @array\n";

my %hash = (
    'key1' => 'value1', 
    'key2' => 'value2'
);

print "hash[1] = $hash{"key1"}\n";