
#= Raytracer in Julia =#

#==============================================================
Vectors
==============================================================#

immutable Vec3
  x::Float64
  y::Float64
  z::Float64
end

Vec3Zero()::Vec3 = Vec3(0,0,0)
vec_sqrt(v::Vec3)::Vec3 = Vec3(sqrt(v.x), sqrt(v.y), sqrt(v.z))
float_eq(a::Float64, b::Float64)::Bool = abs(a - b) < 0.0001
vec_eq(a::Vec3, b::Vec3)::Bool = float_eq(a.x, b.x) && float_eq(a.y, b.y) && float_eq(a.z, b.z)
dot(a::Vec3, b::Vec3)::Float64 = a.x*b.x + a.y*b.y + a.z*b.z
cross(a::Vec3, b::Vec3)::Vec3 = Vec3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x)
lengthSquared(a::Vec3)::Float64 = a.x*a.x + a.y*a.y + a.z*a.z
length(a::Vec3)::Float64 = sqrt(lengthSquared(a))
add(a::Vec3, b::Vec3)::Vec3 = Vec3(a.x+b.x, a.y+b.y, a.z+b.z)
add(a::Vec3, b::Float64)::Vec3 = Vec3(a.x+b, a.y+b, a.z+b)
subtract(a::Vec3, b::Vec3)::Vec3 = Vec3(a.x-b.x, a.y-b.y, a.z-b.z)
mul(a::Vec3, b::Vec3)::Vec3 = Vec3(a.x*b.x, a.y*b.y, a.z*b.z)
mul(a::Vec3, b::Float64)::Vec3 = Vec3(a.x*b, a.y*b, a.z*b)
mul(a::Float64, b::Vec3)::Vec3 = Vec3(a*b.x, a*b.y, a*b.z)
div(a::Vec3, b::Vec3)::Vec3 = Vec3(a.x/b.x, a.y/b.y, a.z/b.z)
div(a::Vec3, b::Float64)::Vec3 = Vec3(a.x/b, a.y/b, a.z/b)
eq(a::Vec3, b::Vec3)::Bool = a.x==b.x && a.y==b.x && a.z==b.z
negate(a::Vec3)::Vec3 = Vec3(-a.x, -a.y, -a.z)
unit_vector(a::Vec3)::Vec3 = div(a, length(a))

function normalized(a::Vec3)::Vec3
  l_squared = lengthSquared(a)
  if float_eq(l_squared, 0) || float_eq(l_squared, 1)
    return a
  end
  return div(a, sqrt(l_squared))
end

#==============================================================
Rays
==============================================================#
immutable Ray
  origin::Vec3
  direction::Vec3
end
pointAtParameter(ray::Ray, t::Float64)::Vec3 = add(ray.origin, mul(ray.direction, t))

#==============================================================
Utils
==============================================================#

function randomInUnitSphere()::Vec3
  p = Vec3(typemax(Float64), typemax(Float64), typemax(Float64))
  while lengthSquared(p) >= 1
    p = subtract(mul(2.0, Vec3(rand(), rand(), rand())), Vec3(1,1,1))
  end; p
end

function reflect(v::Vec3, n::Vec3)::Vec3
  subtract(v, mul(2.0*dot(v, n),n))
end

# This function either returns true and a refracted vector
# or false and Vec3Zero
function refract(v::Vec3, n::Vec3, ni_over_nt::Float64)::Tuple{Bool, Vec3}
  uv = unit_vector(v)
  dt = dot(uv, n)
  discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt)
  if discriminant > 0
    refracted = subtract(mul(ni_over_nt, subtract(uv, mul(n, dt))), mul(n, sqrt(discriminant)))
    return (true, refracted)
  end
  return (false, Vec3Zero())
end

# Polynomial approximation
function schlick(cosine::Float64, reflectiveIndex::Float64)::Float64
  r = ((1-reflectiveIndex) / (1+reflectiveIndex))^2
  r+(1-r)*(1-cosine)^5
end

#==============================================================
Image output
==============================================================#

function writePixelArrayToFile(pixels::Array{Vec3, 2})
  open("output.ppm", "w") do f
    width = size(pixels, 1)
    height = size(pixels, 2)
    write(f, "P3\n$width $height\n255\n")
    for y = reverse(1:height), x = 1:width
        ir = convert(Int64, round(255 * pixels[x, y].x))
        ig = convert(Int64, round(255 * pixels[x, y].y))
        ib = convert(Int64, round(255 * pixels[x, y].z))
        write(f, "$ir $ig $ib\n")
    end
  end
end

#==============================================================
Hitable & HitRecord
==============================================================#
immutable HitRecord
  t::Float64
  p::Vec3
  normal::Vec3
  material::Any
end
abstract Hitable

#==============================================================
Materials
==============================================================#
abstract Material
immutable Lambertian <: Material
  albedo::Vec3
end

immutable Metal <: Material
  albedo::Vec3
  fuzz::Float64
  Metal(albedo::Vec3, fuzz::Float64) = new(albedo, fuzz < 1 ? fuzz : 1)
end

immutable Dialetric <: Material
  reflectiveIndex::Float64
end

function scatter(material::Lambertian, ray::Ray, hitRecord::HitRecord)::Tuple{Bool, Vec3, Ray}
  target = add(add(hitRecord.p, hitRecord.normal), randomInUnitSphere())
  attenuation = material.albedo
  scattered = Ray(hitRecord.p, subtract(target, hitRecord.p))
  return (true, attenuation, scattered)
end

function scatter(material::Metal, ray::Ray, hitRecord::HitRecord)::Tuple{Bool, Vec3, Ray}
  reflected = reflect(unit_vector(ray.direction), hitRecord.normal)
  scattered = Ray(hitRecord.p, add(reflected, mul(material.fuzz, randomInUnitSphere())))
  attenuation = material.albedo
  result = dot(scattered.direction, hitRecord.normal) > 0
  return (result, attenuation, scattered)
end

function scatter(material::Dialetric, ray::Ray, hitRecord::HitRecord)::Tuple{Bool, Vec3, Ray}
  ni_over_nt, reflectProb, cosine = 0.0, 0.0, 0.0
  outwardNormal, refracted = Vec3Zero(), Vec3Zero()
  reflected = reflect(ray.direction, hitRecord.normal)
  attenuation = Vec3(1,1,1)
  scattered = Ray(Vec3Zero(), Vec3Zero())

  if dot(ray.direction, hitRecord.normal) > 0
    outwardNormal = negate(hitRecord.normal)
    ni_over_nt = material.reflectiveIndex
    cosine = material.reflectiveIndex * dot(ray.direction, hitRecord.normal) / length(ray.direction)
  else
    outwardNormal = hitRecord.normal
    ni_over_nt = 1.0 / material.reflectiveIndex
    cosine = -dot(ray.direction, hitRecord.normal) / length(ray.direction)
  end
  result, refracted = refract(ray.direction, outwardNormal, ni_over_nt)
  reflectProb = result ? schlick(cosine, material.reflectiveIndex) : 1
  scattered = (rand() < reflectProb) ? Ray(hitRecord.p, reflected) : Ray(hitRecord.p, refracted)
  return (true, attenuation, scattered)
end

#==============================================================
Geometry
==============================================================#
immutable Sphere <: Hitable
  center::Vec3
  radius::Float64
  material::Material
end

# Specialized Sphere hitable
# Returns result, hitRecord
function hit(hitable::Sphere, ray::Ray, t_min::Float64, t_max::Float64, hitRecord)::Tuple{Bool, Any}
  oc = subtract(ray.origin, hitable.center)         # Vec3
  a = dot(ray.direction, ray.direction)             # Float64
  b = dot(oc, ray.direction)                        # Float64
  c = dot(oc, oc) - hitable.radius * hitable.radius # Float64
  discriminant = b*b - a*c                          # Float64
  if discriminant > 0
    temp = (-b - sqrt(b*b-a*c)) / a
    if temp < t_max && temp > t_min
      t = temp
      p = pointAtParameter(ray, t)
      normal = div(subtract(p, hitable.center), hitable.radius)
      return (true, HitRecord(t, p, normal, hitable.material))
    end

    temp = (-b + sqrt(b*b-a*c)) / a
    if temp < t_max && temp > t_min
      t = temp
      p = pointAtParameter(ray, t)
      normal = div(subtract(p, hitable.center), hitable.radius)
      return (true, HitRecord(t, p, normal, hitable.material))
    end
  end
  return (false, nothing)
end

immutable HitableList <: Hitable
  list::Array{Hitable}
end

# Specialized hitable list hit function
# Returns result, hitRecord
function hit(hitable::HitableList, ray::Ray, t_min::Float64, t_max::Float64, hitRecord)::Tuple{Bool, Any}
    hitAnything = false
    closestSoFar = t_max
    tempHitRecord, hitRecordResult = nothing, nothing
    for element in hitable.list
      result, tempHitRecord = hit(element, ray, t_min, closestSoFar, hitRecord)
      if result
        hitAnything = true
        closestSoFar = tempHitRecord.t
        hitRecordResult = tempHitRecord
      end
    end
    return (hitAnything, hitRecordResult)
end

#==============================================================
Camera
==============================================================#

type Camera
  origin::Vec3
  lowerLeftCorner::Vec3
  horizontal::Vec3
  vertical::Vec3
  u::Vec3
  v::Vec3
  w::Vec3
  lensRadius::Float64

  Camera(lookFrom::Vec3, lookAt::Vec3, viewUp::Vec3, verticalFOV::Float64, aspect::Float64, aperture::Float64, focusDistance::Float64) = new(
    origin,
    lowerLeftCorner,
    horizontal,
    vertical,
    u, v, w, lensRadius)

  # Convenience constructor that takes the tuple produced by _Camera method
  Camera(t::Tuple{Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Float64}) = new(t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8])
end

# Helper methods for Camera constructor
halfHeight(vfov::Float64)::Float64 = tan((vfov*pi/180)/2)
halfWidth(aspect::Float64, half_height::Float64)::Float64 = aspect * half_height
function _Camera(lookFrom::Vec3, lookAt::Vec3, vup::Vec3, vfov::Float64, aspect::Float64, aperture::Float64, focusDistance::Float64)
    lensRadius = aperture / 2
    half_height = halfHeight(vfov)
    half_width = halfWidth(aspect, half_height)
    origin = lookFrom
    w = unit_vector(subtract(lookFrom, lookAt))
    u = unit_vector(cross(vup, w))
    v = cross(w, u)
    a = mul((half_width*focusDistance), u)
    b = mul((half_height*focusDistance), v)
    c = mul(focusDistance, w)
    lowerLeftCorner = subtract(subtract(subtract(origin, a), b), c)
    horizontal = mul(2.0,mul(half_width,mul(focusDistance,u)))
    vertical = mul(2.0,mul(half_height,mul(focusDistance, v)))
    return origin, lowerLeftCorner, horizontal, vertical, u, v, w, lensRadius
end

function randomInUnitDisk()::Vec3
  p = Vec3(typemax(Float64), typemax(Float64), typemax(Float64))
  while dot(p, p) >= 1
    p = subtract(mul(2.0, Vec3(rand(), rand(), 0)), Vec3(1,1,0))
  end
  p
end

function getRay(camera::Camera, s::Float64, t::Float64)::Ray
  rd = mul(camera.lensRadius, randomInUnitDisk())
  offset = mul(camera.u, rd.x * rd.y)
  direction = subtract(subtract(add(add(camera.lowerLeftCorner, mul(s, camera.horizontal)), mul(t, camera.vertical)), camera.origin), offset)
  Ray(add(camera.origin, offset), direction)
end

#==============================================================
Rendering
==============================================================#

function colorFromRay(ray::Ray, world::Hitable, depth::Float64)::Vec3
  hitRecord = nothing
  result, hitRecord = hit(world, ray, 0.001, typemax(Float64), hitRecord)
  if result
    scatterResult, attenuation, scattered = scatter(hitRecord.material, ray, hitRecord)
    if depth < 50 && scatterResult
      return mul(attenuation, colorFromRay(scattered, world, depth+1))
    else
      return Vec3Zero()
    end
  else
    unit_direction = unit_vector(ray.direction)
    t = 0.5 * (unit_direction.y + 1)
    return add(mul((1.0 - t), Vec3(1,1,1)), mul(t, Vec3(0.5, 0.7, 1.0)))
  end
end

function scene()::HitableList
  list = Array{Hitable,1}()
  push!(list, Sphere(Vec3(0,-1000,0), 1000, Lambertian(Vec3(0.5, 0.5, 0.5))))

  for a in -11:10
    for b in -11:10
      chooseMat = rand()
      center = Vec3(Float64(a)+0.9*rand(), 0.2, Float64(b)+0.9*rand())
      if length(subtract(center, Vec3(4,0.2,0))) > 0.9
        # diffuse
        if chooseMat < 0.8
          push!(list, Sphere(center, 0.2, Lambertian(Vec3(rand()*rand(), rand()*rand(), rand()*rand()))))
        # metal
        elseif chooseMat < 0.95
            push!(list, Sphere(center, 0.2, Metal(Vec3(0.5*(1+rand()), 0.5*(1+rand()), 0.5*(1+rand())), 0.5*rand())))
        # glass
        else
          push!(list, Sphere(center, 0.2, Dialetric(1.5)))
        end
      end
    end
  end
  push!(list, Sphere(Vec3(0,1,0), 1.0, Dialetric(1.5)))
  push!(list, Sphere(Vec3(-4,1,0), 1.0, Lambertian(Vec3(0.4,0.2,0.1))))
  push!(list, Sphere(Vec3(4,1,0), 1.0, Metal(Vec3(0.7,0.6,0.5),0.0)))
  HitableList(list)
end

function main()

  # Seed random generator
  srand(0)

  width = 200
  height = 100
  samples = 100
  pixelArray = fill(Vec3Zero(), width, height)
  world = scene()

  lookFrom = Vec3(13,2,3)
  lookAt = Vec3(0,0,0)
  distToFocus = Float64(10)
  aperture = Float64(0.1)
  camera = Camera(_Camera(lookFrom, lookAt, Vec3(0,1,0), Float64(20), Float64(width) / Float64(height), aperture, distToFocus))

  for j = reverse(1:height), i = 1:width
    color = Vec3Zero()
    for sample = 1:samples
      u = Float64(i + rand()) / Float64(width)
      v = Float64(j + rand()) / Float64(height)
      ray = getRay(camera, u, v)
      color = add(color, colorFromRay(ray, world, 0.0))
    end
    color = div(color, Float64(samples))
    color = vec_sqrt(color) # Gamma correction
    pixelArray[i, j] = color
  end
  writePixelArrayToFile(pixelArray)
end
@time main()
