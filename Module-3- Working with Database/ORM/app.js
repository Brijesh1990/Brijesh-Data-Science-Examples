const express=require("express");
const mongoose=require("mongoose");
const bodyParser=require("body-parser");
const Item=require("./models/item");
const app=express();
// middleware ...
app.use(bodyParser.urlencoded({extended:true}));
// Create connection app through mongo DB........
// mongoose.connect('mongodb://localhost:27017/crudApp', { useNewUrlParser: true, useUnifiedTopology: true })
// .then(() => console.log('Connected to MongoDB'))
// .catch(err => console.log('Failed to connect to MongoDB', err));


mongoose.connect('mongodb://127.0.0.1:27017/crudApp', { useNewUrlParser: true, useUnifiedTopology: true })
.then(() => console.log('Connected to MongoDB'))
.catch(err => console.log('Failed to connect to MongoDB', err));mongoose.connect('mongodb://localhost:27017/crudApp', { useNewUrlParser: true, useUnifiedTopology: true })
.then(() => console.log('Connected to MongoDB'))
.catch(err => console.log('Failed to connect to MongoDB', err));

app.set('view engine','ejs');
// Routing for Home page

// list all added items 
app.get('/',async(req,res)=>{
const items=await Item.find();
res.render('index',{items});
});

app.get('/additems',async(req,res)=>{
res.render('new');
});


// Routing for new add data page
app.post('/additems',async(req,res)=>{
// add data in mongoDb database 
const{name,email,phone,pin,age,address}=req.body;
const newItem=new Item({name,email,phone,pin,age,address});
// insert a newItem data in database 
await newItem.save();
res.redirect('/');
});

// Routing for edit page
app.get('/edit/:id',async(req,res)=>{
const item=await Item.findById(req.params.id);
res.render('edit',{item});
});

// Routing for update data page
app.post('/edit/:id',async(req,res)=>{
const{name,email,phone,pin,age,address}=req.body;
await Item.findByIdAndUpdate(req.params.id,{name,email,phone,pin,age,address});
res.redirect('/');

});
// Routing delete data page
app.get('/delete/:id',async(req,res)=>{
await Item.findByIdAndDelete(req.params.id);
res.redirect('/');
});


// create a server 
app.listen(3000,()=>{
console.log('Server running on http://localhost:3000');
});

